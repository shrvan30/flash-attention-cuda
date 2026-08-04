#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cmath>

#define TILE_SIZE 32

#define CHECK_CUDA(call)                                      \
{                                                              \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess)                                   \
    {                                                          \
        std::cerr << "CUDA Error: "                            \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                          \
}

// ------------------------------------------------------------
// Warp-level reductions (inline device helpers)
// ------------------------------------------------------------

static __device__ __forceinline__ float warp_reduce_max(float val)
{
    float reduced_val = val;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        reduced_val = max(
            reduced_val,
            __shfl_xor_sync(0xffffffff, reduced_val, offset)
        );
    }
    // Return thread-local row max to keep rows independent and correct
    return val;
}

static __device__ __forceinline__ float warp_reduce_sum(float val)
{
    float reduced_val = val;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        reduced_val += __shfl_xor_sync(0xffffffff, reduced_val, offset);
    }
    // Return thread-local row sum to keep rows independent and correct
    return val;
}

// ------------------------------------------------------------
// Fused FlashAttention-style Kernel
// ------------------------------------------------------------

__global__ void fused_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d,
    float scale)
{
    int block_row = blockIdx.x;
    int tid = threadIdx.x;
    int row = block_row * TILE_SIZE + tid;

    // Allocate shared memory for K and V tiles only
    __shared__ float Ks[TILE_SIZE][64];
    __shared__ float Vs[TILE_SIZE][64];
    
    // Allocate shared memory for cooperative Q loading
    __shared__ float Qs_shared[TILE_SIZE][64];

    // Local register storage for Q (to prevent bank conflicts and memory latency)
    float Q_local[64];
    for (int k = 0; k < d; k++) {
        Q_local[k] = 0.0f;
    }

    // Cooperative coalesced loading of Q tile into shared memory
    for (int i = 0; i < 64; i++) {
        int flat_idx = i * 32 + tid;
        int r = flat_idx / 64;
        int c = flat_idx % 64;
        int global_row = block_row * TILE_SIZE + r;
        if (global_row < N) {
            Qs_shared[r][c] = Q[global_row * d + c];
        } else {
            Qs_shared[r][c] = 0.0f;
        }
    }
    __syncthreads();

    // Cache Q tile in registers (fast register space access with zero bank conflicts)
    if (row < N) {
        for (int k = 0; k < d; k++) {
            Q_local[k] = Qs_shared[tid][k];
        }
    }
    __syncthreads(); // Synchronize to safely allow K/V loading next

    // Running softmax statistics
    float m_i = -1e20f;
    float l_i = 0.0f;
    float O_i[64];
    for (int k = 0; k < d; k++) {
        O_i[k] = 0.0f;
    }

    int num_col_blocks = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Stream through K/V tiles
    for (int cb = 0; cb < num_col_blocks; cb++)
    {
        int col_start = cb * TILE_SIZE;

        // Cooperative coalesced loading of K/V tiles into shared memory
        for (int i = 0; i < 64; i++) {
            int flat_idx = i * 32 + tid;
            int r = flat_idx / 64;
            int c = flat_idx % 64;
            int global_row = col_start + r;
            if (global_row < N) {
                Ks[r][c] = K[global_row * d + c];
                Vs[r][c] = V[global_row * d + c];
            } else {
                Ks[r][c] = 0.0f;
                Vs[r][c] = 0.0f;
            }
        }
        __syncthreads();

        // Fused compute block
        if (row < N)
        {
            float scores[TILE_SIZE];
            float m_tilde = -1e20f;
            int actual_cols = min(TILE_SIZE, N - col_start);

            // Compute QK^T and keep track of max element
            for (int j = 0; j < actual_cols; j++)
            {
                float dot = 0.0f;
                for (int k = 0; k < d; k++)
                {
                    dot += Q_local[k] * Ks[j][k];
                }
                dot *= scale;
                scores[j] = dot;
                m_tilde = max(m_tilde, dot);
            }
            m_tilde = warp_reduce_max(m_tilde);

            // Softmax exponentiation and local sum
            float l_tilde = 0.0f;
            for (int j = 0; j < actual_cols; j++)
            {
                scores[j] = expf(scores[j] - m_tilde);
                l_tilde += scores[j];
            }
            l_tilde = warp_reduce_sum(l_tilde);

            // Stable online softmax statistics update
            float m_new = max(m_i, m_tilde);
            float l_new = expf(m_i - m_new) * l_i + expf(m_tilde - m_new) * l_tilde;

            float rescale_old = expf(m_i - m_new);
            float rescale_new = expf(m_tilde - m_new);

            // In-place scaling factors computed outside the loop (saves 62 costly divisions per tile!)
            float scale_old = (l_i * rescale_old) / l_new;
            float scale_new = rescale_new / l_new;

            // Immediately accumulate PV into O_i registers
            for (int k = 0; k < d; k++)
            {
                float pv = 0.0f;
                for (int j = 0; j < actual_cols; j++)
                {
                    pv += scores[j] * Vs[j][k];
                }
                O_i[k] = scale_old * O_i[k] + scale_new * pv;
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // Write final output once to global memory (minimal HBM traffic)
    if (row < N)
    {
        for (int k = 0; k < d; k++)
        {
            O[row * d + k] = O_i[k];
        }
    }
}

// ------------------------------------------------------------
// Host launcher
// ------------------------------------------------------------

void launch_fused_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d)
{
    float scale = 1.0f / sqrtf((float)d);

    dim3 threads(TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE);

    fused_attention_kernel<<<blocks, threads>>>(
        d_Q,
        d_K,
        d_V,
        d_O,
        N,
        d,
        scale
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
