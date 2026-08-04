# flash-attention-cuda

Batched multi-head causal FlashAttention for `head_dim = 64`, written from scratch in CUDA and
packaged as a PyTorch extension. fp16 in and out, fp32 accumulation, separate kernels for
prefill and for decoding against a KV cache.

```python
import flashattn_cuda

o = flashattn_cuda.prefill(q, k, v, causal=True, scale=1/8)   # (B, H, N, 64) fp16
o = flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, scale=1/8)
```

## Numbers

RTX 3090 (sm_86, 82 SMs), driver 595.71.05, CUDA 13.2, PyTorch 2.13.0+cu130,
flash-attn 2.8.3.post1. Prefill, **B=8, H=12, causal**, median of three runs of 30 launches:

| N | v2 prefill | torch SDPA (flash) | flash-attn 2.8.3 | v1 fused (scaled to B·H) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1.04 ms · 12.36 TFLOP/s | 0.28 ms · 45.96 | 0.28 ms · 45.98 | 217.8 ms · 0.12 |
| 2048 | 3.88 ms · 13.28 TFLOP/s | 0.92 ms · 55.80 | 0.95 ms · 54.21 | 424.0 ms · 0.24 |
| 4096 | 14.97 ms · 13.77 TFLOP/s | 3.37 ms · 61.20 | 3.49 ms · 59.15 | 888.2 ms · 0.46 |

Decode, single sequence, H=12, one token per call:

| context | v2 decode | torch SDPA reference |
| ---: | ---: | ---: |
| 128 | 0.0123 ms · 81,614 tok/s | 0.0146 ms · 68,611 tok/s |
| 512 | 0.0368 ms · 27,152 tok/s | 0.0199 ms · 50,169 tok/s |
| 2048 | 0.0616 ms · 16,240 tok/s | 0.0213 ms · 46,950 tok/s |

![Prefill throughput, B=8, causal](docs/charts/prefill_b8_causal.svg)

### The honest gap

**v2 prefill is about 4.4x slower than flash-attn**, and 59x faster than the v1 kernels it
replaces at N=4096 (208x at N=1024, where v1's lack of batching hurts most). Two shapes go the
other way: at N=128 v2 is 2–4x *faster* than either library
(0.017 ms vs 0.036 / 0.061 ms at B=1), because at that size both are dominated by fixed launch
overhead that a simpler kernel does not pay. `torch.sdpa`'s flash backend and the `flash-attn`
package track each other closely, as expected — SDPA dispatches to the same FlashAttention-2
kernels.

The gap is not a tuning gap, and the profile says where it comes from. v2 sustains 13.2 TFLOP/s,
which is **37% of the RTX 3090's 35.6 TFLOP/s fp32 FMA peak** — a respectable fraction of the
machine this kernel is using. The problem is which machine it uses: every multiply happens on
the CUDA cores, and the operands arrive through shared memory at roughly 1.75 bytes per FMA,
while an SM sustains 1.0 byte per FMA. That caps this design near 57% of fp32 peak before any
other overhead. flash-attn is not winning by being 4x better at the same game — it runs the
matmuls on the tensor cores, whose fp16 peak is 71 TFLOP/s (2x the fp32 pipe) and which take
their operands from registers via `ldmatrix`/`mma` instead of the shared-memory path that limits
us. 59 TFLOP/s is 83% of *that* ceiling.

So closing the gap means writing a WMMA/MMA kernel, not shaving instructions off this one. That
is deliberately out of scope for v2.0.0 and is the next planned step.

**Decode is parallelism-starved at small batch**, and the benchmark shows it: v2 wins at
context 128 and loses beyond about 256. With the split size fixed at `Sk = 512`, a single
sequence at context 1024 launches `2 splits x 12 heads = 24 blocks` on an 82-SM card, so most of
the GPU is idle and there are nowhere near enough memory requests in flight to reach peak
bandwidth — the measured 83.7 GB/s is 8.9% of the card's 936 GB/s. The per-warp access pattern
is already one fully coalesced 128-byte request per cached row, so the fix is the grid shape
(a smaller `Sk`, or more sequences in the batch), not the inner loop. Full analysis in
[docs/profiles/summary.md](docs/profiles/summary.md).

## What is in v2

| | |
| :-- | :-- |
| shapes | `(B, H, N, 64)`, contiguous row-major, fp16 |
| accumulation | fp32 throughout (scores, softmax statistics, output) |
| prefill | one block per `(batch, head, query tile)`, Br=64 x Bc=32, online softmax |
| causal | key tiles above the diagonal are skipped, not masked |
| decode | split-K over the cached keys, log-sum-exp merge of the partials |
| ragged batches | `seq_lens` is per-sequence; chunks past a sequence's end drop out |
| verification | 86 tests against an fp32 SDPA reference, tolerance 2e-3 |

`head_dim = 64` is a static assertion, not a runtime check: the tile shapes, register blocking
and vectorised loads are all resolved at compile time from it.

## Install

Requires an NVIDIA GPU of compute capability 8.0 or newer, a CUDA toolkit, and a working
PyTorch install.

```bash
git clone https://github.com/shrvan30/flash-attention-cuda.git
cd flash-attention-cuda
pip install -e .
```

The build targets `sm_86` (RTX 3090) by default; set `TORCH_CUDA_ARCH_LIST` for another card:

```bash
TORCH_CUDA_ARCH_LIST="8.0" pip install -e .
```

The extension links against the PyTorch already installed in the environment — an in-tree PEP
517 backend makes that work under pip's default build isolation, so no extra flags are needed.

## Test

```bash
pip install pytest
python -m pytest -m gpu           # 86 GPU tests
```

The suite compares against `scaled_dot_product_attention` computed in fp32 on upcast inputs, so
the measured difference is the kernel's error and not the reference's. It sweeps
B in {1,4} x H in {8,12,14} x N in {128,512,1024,2048,4096} x causal, plus sequence lengths that
are not tile multiples, decode against the causal prefill output, ragged decode batches, and
cache lengths straddling the split boundary.

## Benchmark and profile

```bash
python bench/run_bench.py                                     # -> docs/benchmarks.md
nsys profile -t cuda -o docs/profiles/prefill python bench/profile_workload.py prefill
```

Full tables, chart sources and the machine description are in
[docs/benchmarks.md](docs/benchmarks.md); the kernel-level analysis is in
[docs/profiles/summary.md](docs/profiles/summary.md).

## Layout

```
csrc/            v2 extension: bindings.cpp, prefill.cu, decode.cu, legacy/
src/legacy/      v1 kernels, unchanged, still built by CMake and callable from Python
tests/           GPU correctness suite
bench/           benchmark and profiling drivers
docs/            generated benchmark tables, charts and profiles
```

## v1 — the earlier progression

v1 (tagged `v1.0.0`) was a teaching sequence — CPU baseline, naive GPU, tiled, fused — and its
numbers are kept in [docs/v1.md](docs/v1.md). It is single-head, fp32, non-causal, has no decode
path and no Python binding, so it cannot serve a transformer layer; the kernels are still built
and are benchmarked alongside v2 above for scale. v2 addresses every one of those limitations
except tensor cores.

## Licence

MIT — see [LICENSE](LICENSE).
