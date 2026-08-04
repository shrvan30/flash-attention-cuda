# FlashAttention v2 benchmarks

Every number below was measured by `bench/run_bench.py` on the machine described
here. Times are the median of three runs, each the mean of 30 launches after 10
warm-up launches. TFLOP/s counts `4 * B * H * N^2 * d` and halves it for causal.

| | |
| :-- | :-- |
| GPU | NVIDIA GeForce RTX 3090 (sm_86, 24 GB, 82 SMs) |
| Driver | 595.71.05 |
| CUDA toolkit (nvcc) | 13.2 |
| CUDA runtime (torch) | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Python | 3.12.13 |
| flash-attn | 2.8.3.post1 |

## Prefill — B=1, H=12, causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 12, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.017 | 0.031 | 0.060 | 3.833 | 4.332 |
| 256 | 0.031 | 0.031 | 0.061 | 7.329 | 8.329 |
| 512 | 0.087 | 0.031 | 0.062 | 14.413 | 16.395 |
| 1024 | 0.182 | 0.082 | 0.084 | 25.305 | 29.108 |
| 2048 | 0.591 | 0.167 | 0.169 | 50.553 | 58.090 |
| 4096 | 2.093 | 0.557 | 0.553 | 109.539 | 126.719 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.46 | 0.80 | 0.42 | 0.01 | 0.01 |
| 256 | 3.22 | 3.22 | 1.66 | 0.03 | 0.02 |
| 512 | 4.63 | 12.95 | 6.51 | 0.06 | 0.05 |
| 1024 | 8.87 | 19.56 | 19.24 | 0.13 | 0.11 |
| 2048 | 10.90 | 38.67 | 38.08 | 0.25 | 0.22 |
| 4096 | 12.31 | 46.24 | 46.57 | 0.47 | 0.41 |

## Prefill — B=1, H=12, non-causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 12, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.015 | 0.030 | 0.061 | 3.446 | 3.894 |
| 256 | 0.028 | 0.031 | 0.061 | 6.574 | 7.499 |
| 512 | 0.087 | 0.032 | 0.063 | 12.950 | 14.759 |
| 1024 | 0.274 | 0.087 | 0.089 | 25.907 | 29.303 |
| 2048 | 1.001 | 0.266 | 0.262 | 50.968 | 58.282 |
| 4096 | 3.798 | 0.830 | 0.841 | 109.138 | 126.722 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.25 | 1.67 | 0.83 | 0.01 | 0.01 |
| 256 | 7.18 | 6.58 | 3.31 | 0.03 | 0.03 |
| 512 | 9.29 | 25.21 | 12.78 | 0.06 | 0.05 |
| 1024 | 11.78 | 37.14 | 36.07 | 0.12 | 0.11 |
| 2048 | 12.88 | 48.51 | 49.20 | 0.25 | 0.22 |
| 4096 | 13.57 | 62.06 | 61.28 | 0.47 | 0.41 |

## Prefill — B=8, H=12, causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 96, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.032 | 0.030 | 0.061 | 27.596 | 30.847 |
| 256 | 0.089 | 0.038 | 0.064 | 54.716 | 61.411 |
| 512 | 0.272 | 0.098 | 0.102 | 110.327 | 119.135 |
| 1024 | 1.056 | 0.280 | 0.282 | 217.330 | 237.345 |
| 2048 | 3.886 | 0.916 | 0.941 | 416.534 | 467.013 |
| 4096 | 14.960 | 3.373 | 3.480 | 885.728 | 1017.161 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.33 | 6.67 | 3.30 | 0.01 | 0.01 |
| 256 | 9.03 | 21.08 | 12.65 | 0.03 | 0.03 |
| 512 | 11.84 | 32.96 | 31.45 | 0.06 | 0.05 |
| 1024 | 12.20 | 46.10 | 45.61 | 0.12 | 0.11 |
| 2048 | 13.26 | 56.27 | 54.77 | 0.25 | 0.22 |
| 4096 | 13.78 | 61.12 | 59.23 | 0.47 | 0.41 |

## Prefill — B=8, H=12, non-causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 96, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.048 | 0.031 | 0.061 | 27.740 | 30.985 |
| 256 | 0.119 | 0.040 | 0.061 | 56.132 | 61.912 |
| 512 | 0.440 | 0.123 | 0.130 | 117.453 | 120.737 |
| 1024 | 1.859 | 0.436 | 0.437 | 219.690 | 237.924 |
| 2048 | 7.308 | 1.580 | 1.637 | 415.854 | 467.553 |
| 4096 | 29.010 | 6.178 | 6.525 | 887.656 | 1016.827 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8.43 | 13.14 | 6.59 | 0.01 | 0.01 |
| 256 | 13.56 | 39.96 | 26.45 | 0.03 | 0.03 |
| 512 | 14.66 | 52.43 | 49.42 | 0.05 | 0.05 |
| 1024 | 13.86 | 59.08 | 59.00 | 0.12 | 0.11 |
| 2048 | 14.11 | 65.23 | 62.97 | 0.25 | 0.22 |
| 4096 | 14.21 | 66.74 | 63.19 | 0.46 | 0.41 |

## Decode — single sequence, H=12

Tokens/s is one decoded token per sequence per call, i.e. `batch / latency`.

**What the baseline is.** The comparison line is a **per-step eager SDPA loop**: one `scaled_dot_product_attention` call per decoded token, over the whole cache, with a single-row query. It is a correctness-preserving reference, not a tuned decode kernel — it does not split the KV dimension, so it leaves on the table exactly the parallelism that the fixed-chunk version of our own kernel used to leave. The serious baselines are flash-attn's dedicated decode path (`flash_attn_with_kvcache`) and FlashDecoding, both of which do split KV; measuring against those is deferred. Read the numbers below as "faster than the obvious PyTorch way to decode", not as a claim about state-of-the-art decode kernels.

| context | v2 decode ms | v2 decode tok/s | per-step eager SDPA loop ms | per-step eager SDPA loop tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0122 | 81,835 | 0.0146 | 68,451 |
| 256 | 0.0124 | 80,708 | 0.0146 | 68,616 |
| 512 | 0.0122 | 81,642 | 0.0194 | 51,488 |
| 1024 | 0.0128 | 78,334 | 0.0191 | 52,316 |
| 2048 | 0.0210 | 47,560 | 0.0213 | 46,950 |

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
