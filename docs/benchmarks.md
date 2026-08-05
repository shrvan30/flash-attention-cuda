# FlashAttention v2 benchmarks

Every number below was measured by `bench/run_bench.py` on the machine described
here. Times are the median of three runs, each the mean of 30 launches after 10
warm-up launches. TFLOP/s counts `4 * B * H * N^2 * d` and halves it for causal.

| | |
| :-- | :-- |
| GPU | NVIDIA GeForce RTX 3090 (sm_86, 24 GB, 82 SMs) |
| Driver | 580.126.09 |
| CUDA toolkit (nvcc) | 13.0 |
| CUDA runtime (torch) | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12.13 |
| flash-attn | 2.8.3 |

## Prefill — B=1, H=12, causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 12, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.019 | 0.056 | 0.095 | 4.152 | 4.529 |
| 256 | 0.033 | 0.051 | 0.075 | 7.791 | 8.684 |
| 512 | 0.090 | 0.051 | 0.077 | 15.276 | 17.089 |
| 1024 | 0.167 | 0.079 | 0.093 | 27.906 | 29.983 |
| 2048 | 0.698 | 0.206 | 0.211 | 54.256 | 59.620 |
| 4096 | 2.450 | 0.655 | 0.679 | 118.700 | 130.194 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.36 | 0.45 | 0.26 | 0.01 | 0.01 |
| 256 | 3.05 | 1.96 | 1.33 | 0.03 | 0.02 |
| 512 | 4.49 | 7.89 | 5.24 | 0.05 | 0.05 |
| 1024 | 9.66 | 20.48 | 17.35 | 0.12 | 0.11 |
| 2048 | 9.23 | 31.32 | 30.49 | 0.24 | 0.22 |
| 4096 | 10.52 | 39.35 | 37.93 | 0.43 | 0.40 |

## Prefill — B=1, H=12, non-causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 12, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.017 | 0.051 | 0.077 | 3.705 | 4.024 |
| 256 | 0.029 | 0.050 | 0.076 | 6.948 | 7.698 |
| 512 | 0.088 | 0.050 | 0.077 | 14.088 | 15.188 |
| 1024 | 0.272 | 0.101 | 0.109 | 28.443 | 30.211 |
| 2048 | 1.173 | 0.314 | 0.309 | 54.635 | 59.514 |
| 4096 | 4.403 | 0.978 | 1.040 | 121.938 | 130.141 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.05 | 0.99 | 0.66 | 0.01 | 0.01 |
| 256 | 6.84 | 4.02 | 2.64 | 0.03 | 0.03 |
| 512 | 9.11 | 16.10 | 10.47 | 0.06 | 0.05 |
| 1024 | 11.84 | 31.91 | 29.63 | 0.11 | 0.11 |
| 2048 | 10.99 | 41.06 | 41.64 | 0.24 | 0.22 |
| 4096 | 11.71 | 52.71 | 49.57 | 0.42 | 0.40 |

## Prefill — B=8, H=12, causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 96, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.034 | 0.051 | 0.077 | 29.629 | 32.735 |
| 256 | 0.093 | 0.052 | 0.076 | 67.102 | 67.083 |
| 512 | 0.274 | 0.136 | 0.165 | 134.323 | 125.603 |
| 1024 | 1.238 | 0.345 | 0.424 | 238.309 | 244.108 |
| 2048 | 4.575 | 1.154 | 1.305 | 438.121 | 476.106 |
| 4096 | 17.644 | 4.230 | 4.805 | 1092.433 | 1077.559 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5.96 | 3.95 | 2.63 | 0.01 | 0.01 |
| 256 | 8.68 | 15.63 | 10.58 | 0.02 | 0.02 |
| 512 | 11.76 | 23.66 | 19.56 | 0.05 | 0.05 |
| 1024 | 10.40 | 37.33 | 30.42 | 0.11 | 0.11 |
| 2048 | 11.26 | 44.66 | 39.49 | 0.24 | 0.22 |
| 4096 | 11.68 | 48.74 | 42.90 | 0.38 | 0.38 |

## Prefill — B=8, H=12, non-causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 96, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.049 | 0.051 | 0.077 | 30.199 | 32.912 |
| 256 | 0.133 | 0.069 | 0.092 | 69.314 | 70.002 |
| 512 | 0.457 | 0.157 | 0.168 | 139.159 | 127.166 |
| 1024 | 2.213 | 0.532 | 0.556 | 271.044 | 246.369 |
| 2048 | 8.661 | 1.935 | 2.089 | 440.913 | 480.215 |
| 4096 | 34.149 | 7.453 | 8.039 | 959.732 | 1044.224 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8.21 | 7.97 | 5.26 | 0.01 | 0.01 |
| 256 | 12.12 | 23.28 | 17.48 | 0.02 | 0.02 |
| 512 | 14.09 | 41.01 | 38.33 | 0.05 | 0.05 |
| 1024 | 11.65 | 48.48 | 46.32 | 0.10 | 0.10 |
| 2048 | 11.90 | 53.27 | 49.34 | 0.23 | 0.21 |
| 4096 | 12.07 | 55.32 | 51.29 | 0.43 | 0.39 |

## Decode — single sequence, H=12

Tokens/s is one decoded token per sequence per call, i.e. `batch / latency`.

**What the baseline is.** The comparison line is a **per-step eager SDPA loop**: one `scaled_dot_product_attention` call per decoded token, over the whole cache, with a single-row query. It is a correctness-preserving reference, not a tuned decode kernel — it does not split the KV dimension, so it leaves on the table exactly the parallelism that the fixed-chunk version of our own kernel used to leave. The serious baselines are flash-attn's dedicated decode path (`flash_attn_with_kvcache`) and FlashDecoding, both of which do split KV; measuring against those is deferred. Read the numbers below as "faster than the obvious PyTorch way to decode", not as a claim about state-of-the-art decode kernels.

| context | v2 decode ms | v2 decode tok/s | per-step eager SDPA loop ms | per-step eager SDPA loop tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0169 | 59,305 | 0.0222 | 45,003 |
| 256 | 0.0172 | 58,014 | 0.0224 | 44,660 |
| 512 | 0.0170 | 58,947 | 0.0303 | 32,992 |
| 1024 | 0.0173 | 57,899 | 0.0300 | 33,368 |
| 2048 | 0.0314 | 31,844 | 0.0302 | 33,066 |

## Charts

### Prefill B=1, causal

![Prefill B=1, causal](charts/prefill_b1_causal.svg)

### Prefill B=1, non-causal

![Prefill B=1, non-causal](charts/prefill_b1_noncausal.svg)

### Prefill B=8, causal

![Prefill B=8, causal](charts/prefill_b8_causal.svg)

### Prefill B=8, non-causal

![Prefill B=8, non-causal](charts/prefill_b8_noncausal.svg)

### Decode tokens/s

![Decode tokens/s](charts/decode_tokens_per_s.svg)
