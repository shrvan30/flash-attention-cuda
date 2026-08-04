#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cmath>

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
// Naive Attention Kernel
// Each thread computes ONE output element O[row, col]
// ------------------------------------------------------------

__global__ void naive_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d,
    float scale)
{
    // --------------------------------------------------------
    // Global thread indices
    // --------------------------------------------------------

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check

    if (row >= N || col >= d)
        return;

    // --------------------------------------------------------
    // Step 1: Find row max for stable softmax
    // --------------------------------------------------------

    float s_max = -1e20f;

    for (int j = 0; j < N; j++)
    {
        float dot = 0.0f;

        for (int k = 0; k < d; k++)
        {
            dot += Q[row * d + k] * K[j * d + k];
        }

        dot *= scale;

        if (dot > s_max)
            s_max = dot;
    }

    // --------------------------------------------------------
    // Step 2: Compute denominator
    // --------------------------------------------------------

    float denom = 0.0f;

    for (int j = 0; j < N; j++)
    {
        float dot = 0.0f;

        for (int k = 0; k < d; k++)
        {
            dot += Q[row * d + k] * K[j * d + k];
        }

        dot *= scale;

        denom += expf(dot - s_max);
    }

    // --------------------------------------------------------
    // Step 3: Compute weighted sum
    // --------------------------------------------------------

    float out = 0.0f;

    for (int j = 0; j < N; j++)
    {
        float dot = 0.0f;

        for (int k = 0; k < d; k++)
        {
            dot += Q[row * d + k] * K[j * d + k];
        }

        dot *= scale;

        float p = expf(dot - s_max) / denom;

        out += p * V[j * d + col];
    }

    // --------------------------------------------------------
    // Store output
    // --------------------------------------------------------

    O[row * d + col] = out;
}


// ------------------------------------------------------------
// Host launcher
// ------------------------------------------------------------

void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d)
{
    float scale = 1.0f / sqrtf((float)d);

    dim3 threads(16, 16);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (d + threads.y - 1) / threads.y
    );

    naive_attention_kernel<<<blocks, threads>>>(
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
