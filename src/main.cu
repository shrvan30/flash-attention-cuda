#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>

// ------------------------------------------------------------
// Kernel Launchers
// ------------------------------------------------------------

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
                dot += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = dot * scale;
        }
    }

    // Softmax
    for (int i = 0; i < N; i++)
    {
        float row_max = -1e20f;
        for (int j = 0; j < N; j++)
        {
            row_max = std::max(row_max, S[i * N + j]);
        }

        float denom = 0.0f;
        for (int j = 0; j < N; j++)
        {
            S[i * N + j] = std::exp(S[i * N + j] - row_max);
            denom += S[i * N + j];
        }

        for (int j = 0; j < N; j++)
        {
            S[i * N + j] /= denom;
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
                out += S[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = out;
        }
    }
}

// ------------------------------------------------------------
// Struct for benchmark metrics
// ------------------------------------------------------------
struct KernelMetric {
    float avg_time_ms;
    float speedup;
    float max_error;
    float gflops;
    float bandwidth_gbs;
};

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main()
{
    const int d = 64;
    const std::vector<int> sequence_lengths = {256, 512, 1024, 2048, 4096};
    const int GPU_ITERATIONS = 15;

    // Open CSV file for writing results
    std::string csv_path = "benchmarks/results/benchmark_results.csv";
    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open " << csv_path << " for writing!\n";
        return 1;
    }

    // CSV Header
    csv_file << "N,d,Kernel,AvgTimeMS,Speedup,MaxError,GFLOPs,BandwidthGBs,MemoryMB\n";

    std::cout << "\n============================================================\n";
    std::cout << "          FLASHATTENTION CUDA BENCHMARKING SUITE\n";
    std::cout << "============================================================\n";
    std::cout << "Testing sequence lengths N = {256, 512, 1024, 2048, 4096} with d = " << d << "\n";
    std::cout << "GPU runs will use " << GPU_ITERATIONS << " iterations after warmup.\n";
    std::cout << "CPU iterations will dynamically scale to keep runtime reasonable.\n";
    std::cout << "------------------------------------------------------------\n\n";

    // Random initialization setup
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int N : sequence_lengths)
    {
        std::cout << ">>> Running benchmark for N = " << N << ", d = " << d << " <<<\n";

        size_t size = N * d * sizeof(float);

        // Host memory allocation
        std::vector<float> h_Q(N * d);
        std::vector<float> h_K(N * d);
        std::vector<float> h_V(N * d);
        std::vector<float> h_O_cpu(N * d);
        std::vector<float> h_O_gpu(N * d);

        // Random fill
        for (int i = 0; i < N * d; i++)
        {
            h_Q[i] = dist(gen);
            h_K[i] = dist(gen);
            h_V[i] = dist(gen);
        }

        // --------------------------------------------------------
        // 1. CPU REFERENCE RUNS
        // --------------------------------------------------------
        int cpu_iterations = 15;
        if (N == 1024) cpu_iterations = 10;
        else if (N == 2048) cpu_iterations = 5;
        else if (N == 4096) cpu_iterations = 3;

        std::cout << "  - CPU Baseline (" << cpu_iterations << " iterations): ";
        std::cout.flush();

        // Warmup CPU
        attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, d);

        float cpu_total_ms = 0.0f;
        for (int iter = 0; iter < cpu_iterations; iter++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, d);
            auto end_time = std::chrono::high_resolution_clock::now();
            cpu_total_ms += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
        float cpu_avg_ms = cpu_total_ms / cpu_iterations;
        std::cout << std::fixed << std::setprecision(3) << cpu_avg_ms << " ms\n";

        // Compute metrics for CPU
        double cpu_flops = 4.0 * N * N * d;
        float cpu_gflops = (cpu_flops / 1e9) / (cpu_avg_ms / 1000.0);
        double std_mem_bytes = 2.0 * N * N * sizeof(float);
        float std_mem_mb = std_mem_bytes / (1024.0 * 1024.0);
        float cpu_bw = (16.0f * N * d) / (cpu_avg_ms * 1e6f); // algorithmic min transfer

        // Write CPU to CSV
        csv_file << N << "," << d << ",CPU," << cpu_avg_ms << ",1.0,0.0," << cpu_gflops << "," << cpu_bw << "," << std_mem_mb << "\n";

        // --------------------------------------------------------
        // Device Memory Allocation
        // --------------------------------------------------------
        float *d_Q, *d_K, *d_V, *d_O;
        cudaMalloc(&d_Q, size);
        cudaMalloc(&d_K, size);
        cudaMalloc(&d_V, size);
        cudaMalloc(&d_O, size);

        cudaMemcpy(d_Q, h_Q.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), size, cudaMemcpyHostToDevice);

        // GPU timing events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Define a helper lambda for timing GPU kernels
        auto benchmark_gpu_kernel = [&](const std::string& name, auto launcher) -> KernelMetric {
            std::cout << "  - " << name << " Kernel (" << GPU_ITERATIONS << " iterations): ";
            std::cout.flush();

            // A. Warmup launch
            launcher(d_Q, d_K, d_V, d_O, N, d);
            cudaDeviceSynchronize();

            // B. Timing loop
            float total_ms = 0.0f;
            for (int iter = 0; iter < GPU_ITERATIONS; iter++) {
                cudaMemset(d_O, 0, size); // Reset output buffer to avoid stale state
                cudaEventRecord(start);
                launcher(d_Q, d_K, d_V, d_O, N, d);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }
            float avg_ms = total_ms / GPU_ITERATIONS;
            std::cout << avg_ms << " ms";

            // C. Correctness verification
            cudaMemcpy(h_O_gpu.data(), d_O, size, cudaMemcpyDeviceToHost);
            float max_err = 0.0f;
            for (int i = 0; i < N * d; i++) {
                max_err = std::max(max_err, std::abs(h_O_cpu[i] - h_O_gpu[i]));
            }
            std::cout << " | Max Error: " << std::scientific << max_err << std::fixed << std::setprecision(3);

            // D. Compute stats
            float speedup = cpu_avg_ms / avg_ms;
            float gflops = (cpu_flops / 1e9) / (avg_ms / 1000.0);
            float bw = (16.0f * N * d) / (avg_ms * 1e6f); // algorithmic min transfer
            
            std::cout << " | Speedup: " << speedup << "x | GFLOPs: " << gflops << "\n";

            // Write to CSV
            csv_file << N << "," << d << "," << name << "," << avg_ms << "," << speedup << "," 
                     << max_err << "," << gflops << "," << bw << "," << (name == "Fused" ? 0.0f : std_mem_mb) << "\n";

            return {avg_ms, speedup, max_err, gflops, bw};
        };

        // 2. NAIVE CUDA BENCHMARK
        benchmark_gpu_kernel("Naive", launch_naive_attention);

        // 3. TILED CUDA BENCHMARK
        benchmark_gpu_kernel("Tiled", launch_tiled_attention);

        // 4. FUSED FLASHATTENTION BENCHMARK
        benchmark_gpu_kernel("Fused", launch_fused_attention);

        std::cout << "\n";

        // Cleanup configuration memory
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
    }

    csv_file.close();
    std::cout << "============================================================\n";
    std::cout << "Benchmarking phase complete! Results successfully saved to:\n";
    std::cout << "  " << csv_path << "\n";
    std::cout << "============================================================\n\n";

    return 0;
}
