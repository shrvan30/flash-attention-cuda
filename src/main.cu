#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d);

void launch_tiled_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d);

void launch_fused_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d);


// ------------------------------------------------------------
// CPU Reference
// ------------------------------------------------------------

void attention_cpu(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int N,
    int d)
{
    float scale = 1.0f / std::sqrt((float)d);

    std::vector<float> S(N * N);

    // QK^T

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float dot = 0.0f;

            for (int k = 0; k < d; k++)
            {
                dot += Q[i*d+k] * K[j*d+k];
            }

            S[i*N+j] = dot * scale;
        }
    }

    // softmax

    for (int i = 0; i < N; i++)
    {
        float row_max = -1e20f;

        for (int j = 0; j < N; j++)
        {
            row_max = std::max(row_max, S[i*N+j]);
        }

        float denom = 0.0f;

        for (int j = 0; j < N; j++)
        {
            S[i*N+j] = expf(S[i*N+j] - row_max);

            denom += S[i*N+j];
        }

        for (int j = 0; j < N; j++)
        {
            S[i*N+j] /= denom;
        }
    }

    // PV

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d; j++)
        {
            float out = 0.0f;

            for (int k = 0; k < N; k++)
            {
                out += S[i*N+k] * V[k*d+j];
            }

            O[i*d+j] = out;
        }
    }
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    int N = 512;

    int d = 64;

    size_t size = N * d * sizeof(float);

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<float> h_Q(N * d);

    std::vector<float> h_K(N * d);

    std::vector<float> h_V(N * d);

    std::vector<float> h_O_cpu(N * d);

    std::vector<float> h_O_gpu(N * d);

    // --------------------------------------------------------
    // Random initialization
    // --------------------------------------------------------

    std::mt19937 gen(42);

    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N * d; i++)
    {
        h_Q[i] = dist(gen);

        h_K[i] = dist(gen);

        h_V[i] = dist(gen);
    }

    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    std::cout << "\nRunning CPU baseline...\n";

    auto cpu_start = std::chrono::high_resolution_clock::now();

    attention_cpu(
        h_Q,
        h_K,
        h_V,
        h_O_cpu,
        N,
        d
    );

    auto cpu_end = std::chrono::high_resolution_clock::now();

    float cpu_ms =
        std::chrono::duration<float, std::milli>(
            cpu_end - cpu_start
        ).count();

    std::cout << "CPU Time: "
              << cpu_ms
              << " ms\n";

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float *d_Q, *d_K, *d_V, *d_O;

    cudaMalloc(&d_Q, size);

    cudaMalloc(&d_K, size);

    cudaMalloc(&d_V, size);

    cudaMalloc(&d_O, size);

    // --------------------------------------------------------
    // Copy to device
    // --------------------------------------------------------

    cudaMemcpy(
        d_Q,
        h_Q.data(),
        size,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_K,
        h_K.data(),
        size,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_V,
        h_V.data(),
        size,
        cudaMemcpyHostToDevice
    );

    // --------------------------------------------------------
    // GPU timing
    // --------------------------------------------------------

    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventRecord(start);

    launch_naive_attention(
        d_Q,
        d_K,
        d_V,
        d_O,
        N,
        d
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;

    cudaEventElapsedTime(&gpu_ms, start, stop);

    std::cout << "GPU Time: "
              << gpu_ms
              << " ms\n";

    // --------------------------------------------------------
    // Copy back
    // --------------------------------------------------------

    cudaMemcpy(
        h_O_gpu.data(),
        d_O,
        size,
        cudaMemcpyDeviceToHost
    );

    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    float max_error = 0.0f;

    for (int i = 0; i < N * d; i++)
    {
        max_error = std::max(
            max_error,
            fabs(h_O_cpu[i] - h_O_gpu[i])
        );
    }

    std::cout << "Max Error: "
              << max_error
              << std::endl;

    // --------------------------------------------------------
    // Tiled GPU benchmark
    // --------------------------------------------------------

    cudaEventRecord(start);

    launch_tiled_attention(
        d_Q,
        d_K,
        d_V,
        d_O,
        N,
        d
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float tiled_ms = 0.0f;

    cudaEventElapsedTime(
        &tiled_ms,
        start,
        stop
    );

    std::cout << "Tiled GPU Time: "
              << tiled_ms
              << " ms\n";

    // --------------------------------------------------------
    // Copy tiled output
    // --------------------------------------------------------

    cudaMemcpy(
        h_O_gpu.data(),
        d_O,
        size,
        cudaMemcpyDeviceToHost
    );

    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    float tiled_error = 0.0f;

    for (int i = 0; i < N * d; i++)
    {
        tiled_error = std::max(
            tiled_error,
            fabs(h_O_cpu[i] - h_O_gpu[i])
        );
    }

    std::cout << "Tiled Max Error: "
              << tiled_error
              << std::endl;

    // --------------------------------------------------------
    // Fused GPU benchmark
    // --------------------------------------------------------

    cudaEventRecord(start);

    launch_fused_attention(
        d_Q,
        d_K,
        d_V,
        d_O,
        N,
        d
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float fused_ms = 0.0f;

    cudaEventElapsedTime(
        &fused_ms,
        start,
        stop
    );

    std::cout << "Fused GPU Time: "
              << fused_ms
              << " ms\n";

    // --------------------------------------------------------
    // Copy fused output
    // --------------------------------------------------------

    cudaMemcpy(
        h_O_gpu.data(),
        d_O,
        size,
        cudaMemcpyDeviceToHost
    );

    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    float fused_error = 0.0f;

    for (int i = 0; i < N * d; i++)
    {
        fused_error = std::max(
            fused_error,
            fabs(h_O_cpu[i] - h_O_gpu[i])
        );
    }

    std::cout << "Fused Max Error: "
              << fused_error
              << std::endl;

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cudaFree(d_Q);

    cudaFree(d_K);

    cudaFree(d_V);

    cudaFree(d_O);

    return 0;
}
