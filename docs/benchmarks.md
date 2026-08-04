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
| 128 | 0.017 | 0.032 | 0.062 | 3.834 | 4.297 |
| 256 | 0.031 | 0.031 | 0.062 | 7.306 | 8.315 |
| 512 | 0.087 | 0.031 | 0.064 | 14.382 | 16.404 |
| 1024 | 0.182 | 0.082 | 0.084 | 25.338 | 28.945 |
| 2048 | 0.585 | 0.166 | 0.169 | 50.702 | 57.204 |
| 4096 | 2.050 | 0.552 | 0.541 | 108.598 | 125.959 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.46 | 0.79 | 0.41 | 0.01 | 0.01 |
| 256 | 3.21 | 3.22 | 1.64 | 0.03 | 0.02 |
| 512 | 4.64 | 12.84 | 6.28 | 0.06 | 0.05 |
| 1024 | 8.85 | 19.53 | 19.24 | 0.13 | 0.11 |
| 2048 | 11.02 | 38.74 | 38.19 | 0.25 | 0.23 |
| 4096 | 12.57 | 46.72 | 47.67 | 0.47 | 0.41 |

## Prefill — B=1, H=12, non-causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 12, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.015 | 0.031 | 0.062 | 3.431 | 3.845 |
| 256 | 0.028 | 0.031 | 0.062 | 6.564 | 7.437 |
| 512 | 0.087 | 0.031 | 0.061 | 12.858 | 14.668 |
| 1024 | 0.273 | 0.088 | 0.090 | 25.714 | 29.174 |
| 2048 | 0.977 | 0.262 | 0.261 | 50.616 | 57.194 |
| 4096 | 3.773 | 0.849 | 0.842 | 109.864 | 126.855 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.26 | 1.62 | 0.82 | 0.01 | 0.01 |
| 256 | 7.19 | 6.52 | 3.26 | 0.03 | 0.03 |
| 512 | 9.31 | 26.24 | 13.13 | 0.06 | 0.05 |
| 1024 | 11.80 | 36.76 | 35.69 | 0.13 | 0.11 |
| 2048 | 13.18 | 49.26 | 49.40 | 0.25 | 0.23 |
| 4096 | 13.66 | 60.73 | 61.23 | 0.47 | 0.41 |

## Prefill — B=8, H=12, causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 96, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.032 | 0.031 | 0.062 | 27.702 | 31.320 |
| 256 | 0.089 | 0.038 | 0.062 | 53.989 | 61.145 |
| 512 | 0.273 | 0.100 | 0.105 | 112.667 | 118.997 |
| 1024 | 1.032 | 0.278 | 0.281 | 218.343 | 237.994 |
| 2048 | 3.874 | 0.917 | 0.955 | 411.599 | 466.934 |
| 4096 | 14.965 | 3.344 | 3.481 | 879.428 | 1015.746 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.29 | 6.45 | 3.25 | 0.01 | 0.01 |
| 256 | 9.00 | 21.25 | 12.91 | 0.03 | 0.03 |
| 512 | 11.80 | 32.07 | 30.81 | 0.06 | 0.05 |
| 1024 | 12.49 | 46.33 | 45.79 | 0.12 | 0.11 |
| 2048 | 13.31 | 56.18 | 53.97 | 0.25 | 0.22 |
| 4096 | 13.78 | 61.66 | 59.22 | 0.47 | 0.41 |

## Prefill — B=8, H=12, non-causal

The v1 kernels are single-head, fp32 and have no causal path, so they cannot run this shape directly. Their column is one `(N, 64)` launch timed and multiplied by B*H = 96, which is what covering the same total work would cost; the causal tables reuse the same (non-causal) v1 measurement, so v1 is if anything flattered there.

| N | v2 prefill ms | torch SDPA (flash backend) ms | flash-attn (official) ms | v1 fused (fp32, 1 head) ms | v1 tiled (fp32, 1 head) ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.048 | 0.032 | 0.062 | 27.663 | 31.300 |
| 256 | 0.118 | 0.040 | 0.062 | 56.592 | 62.118 |
| 512 | 0.452 | 0.122 | 0.127 | 111.624 | 119.085 |
| 1024 | 1.837 | 0.440 | 0.440 | 218.481 | 238.074 |
| 2048 | 7.312 | 1.569 | 1.637 | 411.236 | 467.780 |
| 4096 | 29.010 | 6.100 | 6.414 | 872.261 | 1007.586 |

| N | v2 prefill TFLOP/s | torch SDPA (flash backend) TFLOP/s | flash-attn (official) TFLOP/s | v1 fused (fp32, 1 head) TFLOP/s | v1 tiled (fp32, 1 head) TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8.46 | 12.74 | 6.54 | 0.01 | 0.01 |
| 256 | 13.61 | 39.99 | 26.01 | 0.03 | 0.03 |
| 512 | 14.27 | 53.00 | 50.71 | 0.06 | 0.05 |
| 1024 | 14.03 | 58.56 | 58.55 | 0.12 | 0.11 |
| 2048 | 14.10 | 65.68 | 62.96 | 0.25 | 0.22 |
| 4096 | 14.21 | 67.59 | 64.29 | 0.47 | 0.41 |

## Decode — single sequence, H=12

Tokens/s is one decoded token per sequence per call, i.e. `batch / latency`.

| context | v2 decode ms | v2 decode tok/s | torch SDPA (decode reference) ms | torch SDPA (decode reference) tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0122 | 82,064 | 0.0143 | 69,754 |
| 256 | 0.0124 | 80,931 | 0.0145 | 68,737 |
| 512 | 0.0122 | 81,642 | 0.0188 | 53,170 |
| 1024 | 0.0127 | 78,755 | 0.0189 | 52,978 |
| 2048 | 0.0210 | 47,637 | 0.0213 | 47,011 |

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
