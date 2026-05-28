import numpy as np
import time
import csv
import os
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def stable_softmax(x):
    """
    Numerically stable softmax.
    
    x: (N, N)
    """
    
    # subtract row-wise max for numerical stability
    x_max = np.max(x, axis=1, keepdims=True)

    x_exp = np.exp(x - x_max)

    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    return x_exp / x_sum

def attention_cpu(Q, K, V, scale=None):
    """
    Standard Attention:
    O = softmax(QK^T / sqrt(d))V
    Q, K, V : (N, d)
    """

    N, d = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    # ---------------------------------------------------
    # Step 1: Compute attention scores
    # S = QK^T
    # Shape: (N, N)
    # ---------------------------------------------------

    S = (Q @ K.T) * scale
    #Here Q @ K.T is equivalent to np.matmul(Q, K.T) & np.matmul(Q, K.T) in 2D.

    # ---------------------------------------------------
    # Step 2: Softmax
    # Shape: (N, N)
    # ---------------------------------------------------

    P = stable_softmax(S)

    # ---------------------------------------------------
    # Step 3: Output
    # O = PV
    # Shape: (N, d)
    # ---------------------------------------------------
    O = P @ V
    return O


def benchmark_attention_cpu(N, d, n_runs=10):
    """
    Benchmark CPU attention implementation.
    """
    print(f"\nRunning CPU Attention Benchmark")
    print(f"N = {N}, d = {d}")
    # ----------------------------------------
    # Random input tensors
    # ----------------------------------------

    Q = np.random.randn(N, d).astype(np.float32)

    K = np.random.randn(N, d).astype(np.float32)

    V = np.random.randn(N, d).astype(np.float32)

    # ----------------------------------------
    # Warmup
    # ----------------------------------------
    _ = attention_cpu(Q, K, V)

    # ----------------------------------------
    # Timing
    # ----------------------------------------
    start = time.perf_counter()
    for _ in range(n_runs):
        O = attention_cpu(Q, K, V)

    end = time.perf_counter()
    avg_time_ms = ((end - start) / n_runs) * 1000

    # ----------------------------------------
    # Memory estimation
    # ----------------------------------------

    # Memory for Attention score matrix S & Probability matrix P

    memory_bytes = 2 * (N * N * 4)

    memory_mb = memory_bytes / (1024 ** 2)

    print(f"Average Time : {avg_time_ms:.3f} ms")
    print(f"Estimated Memory Usage : {memory_mb:.2f} MB")

    return avg_time_ms, memory_mb


def run_full_benchmark():

    configs = [
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64)
    ]

    results = []

    print("\n" + "=" * 70)
    print("CPU ATTENTION BENCHMARK")
    print("=" * 70)

    for N, d in configs:

        runtime_ms, memory_mb = benchmark_attention_cpu(
            N=N,
            d=d,
            n_runs=10
        )

        results.append([
            N,
            d,
            runtime_ms,
            memory_mb
        ])

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------

    os.makedirs("../benchmarks/result", exist_ok=True)

    csv_path = "../benchmarks/result/cpu_benchmark.csv"

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Sequence_Length",
            "Embedding_Dim",
            "Runtime_ms",
            "Memory_MB"
        ])

        writer.writerows(results)

    print(f"\nBenchmark saved to:")
    print(csv_path)

    # -------------------------------------------------
    # Plot Runtime Graph
    # -------------------------------------------------

    if not HAS_MATPLOTLIB:
        print("\nmatplotlib is not installed. Skipping plot generation.")
        return

    seq_lengths = [r[0] for r in results]

    runtimes = [r[2] for r in results]

    plt.figure(figsize=(8, 5))

    plt.plot(seq_lengths, runtimes, marker='o')

    plt.xlabel("Sequence Length (N)")

    plt.ylabel("Runtime (ms)")

    plt.title("CPU Attention Runtime vs Sequence Length")

    plt.grid(True)

    os.makedirs("../benchmarks/python_cpu_plot", exist_ok=True)

    plot_path = "../benchmarks/python_cpu_plot/cpu_runtime.png"

    plt.savefig(plot_path)

    print(f"Runtime plot saved to:")
    print(plot_path)


if __name__ == "__main__":

    # -------------------------------------------------
    # Step 8: Validation Check
    # -------------------------------------------------
    print("=" * 70)
    print("VALIDATION CHECK")
    print("=" * 70)
    Q_val = np.random.randn(4, 8).astype(np.float32)
    K_val = np.random.randn(4, 8).astype(np.float32)
    V_val = np.random.randn(4, 8).astype(np.float32)
    O_val = attention_cpu(Q_val, K_val, V_val)
    print(f"Validation Output Shape: {O_val.shape} (Expected: (4, 8))")
    print("=" * 70)

    run_full_benchmark()
