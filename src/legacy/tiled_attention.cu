#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cmath>
#include "softmax_stable.cu"

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
// Shared-Memory Tiled Attention Kernel
// FlashAttention-inspired
// ------------------------------------------------------------

__global__ void tiled_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d,
    float scale)
{
    // --------------------------------------------------------
    // Thread indexing
    // --------------------------------------------------------

    int block_row = blockIdx.x;

    int tid = threadIdx.x;

    int row = block_row * TILE_SIZE + tid;

    // --------------------------------------------------------
    // Shared memory tiles
    // --------------------------------------------------------

    __shared__ float Qs[TILE_SIZE][64];

    __shared__ float Ks[TILE_SIZE][64];

    __shared__ float Vs[TILE_SIZE][64];

    // --------------------------------------------------------
    // Running softmax statistics
    // --------------------------------------------------------

    float m_i = -1e20f;

    float l_i = 0.0f;

    float O_i[64];

    for (int k = 0; k < d; k++)
    {
        O_i[k] = 0.0f;
    }

    // --------------------------------------------------------
    // Load Q tile into shared memory
    // --------------------------------------------------------

    if (row < N)
    {
        for (int k = 0; k < d; k++)
        {
            Qs[tid][k] = Q[row * d + k];
        }
    }

    __syncthreads();

    // --------------------------------------------------------
    // Iterate over K/V tiles
    // --------------------------------------------------------

    int num_col_blocks =
        (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int cb = 0; cb < num_col_blocks; cb++)
    {
        int col_start = cb * TILE_SIZE;

        int col_row = col_start + tid;

        // ----------------------------------------------------
        // Load K/V tiles into shared memory
        // ----------------------------------------------------

        if (col_row < N)
        {
            for (int k = 0; k < d; k++)
            {
                Ks[tid][k] = K[col_row * d + k];

                Vs[tid][k] = V[col_row * d + k];
            }
        }

        __syncthreads();

        // ----------------------------------------------------
        // Compute attention block
        // ----------------------------------------------------

        if (row < N)
        {
            float S_block[TILE_SIZE];

            float m_tilde = -1e20f;

            int actual_cols =
                min(TILE_SIZE, N - col_start);

            // ------------------------------------------------
            // Compute QK^T block
            // ------------------------------------------------

            for (int j = 0; j < actual_cols; j++)
            {
                float dot = 0.0f;

                for (int k = 0; k < d; k++)
                {
                    dot += Qs[tid][k] * Ks[j][k];
                }

                dot *= scale;

                S_block[j] = dot;

                m_tilde = max(m_tilde, dot);
            }

            m_tilde = warp_reduce_max(m_tilde);

            // ------------------------------------------------
            // Softmax block
            // ------------------------------------------------

            float P_block[TILE_SIZE];

            float l_tilde = 0.0f;

            for (int j = 0; j < actual_cols; j++)
            {
                P_block[j] =
                    expf(S_block[j] - m_tilde);

                l_tilde += P_block[j];
            }

            l_tilde = warp_reduce_sum(l_tilde);

            // ------------------------------------------------
            // Online softmax update
            // ------------------------------------------------

            float m_new =
                max(m_i, m_tilde);

            float l_new =
                expf(m_i - m_new) * l_i +
                expf(m_tilde - m_new) * l_tilde;

            float rescale_old =
                expf(m_i - m_new);

            float rescale_new =
                expf(m_tilde - m_new);

            // ------------------------------------------------
            // Update output
            // ------------------------------------------------

            for (int k = 0; k < d; k++)
            {
                float pv = 0.0f;

                for (int j = 0; j < actual_cols; j++)
                {
                    pv += P_block[j] * Vs[j][k];
                }

                O_i[k] =
                    (
                        l_i * rescale_old * O_i[k]
                        +
                        rescale_new * pv
                    ) / l_new;
            }

            m_i = m_new;

            l_i = l_new;
        }

        __syncthreads();
    }

    // --------------------------------------------------------
    // Store output
    // --------------------------------------------------------

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

void launch_tiled_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d)
{
    float scale =
        1.0f / sqrtf((float)d);

    dim3 threads(TILE_SIZE);

    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    tiled_attention_kernel<<<blocks, threads>>>(
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
