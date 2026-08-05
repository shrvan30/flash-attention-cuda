# flash-attention-cuda

[![CI](https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml)

Batched multi-head causal FlashAttention for `head_dim = 64`, written from scratch in CUDA and
packaged as a PyTorch extension. fp16 in and out, fp32 accumulation, separate kernels for
prefill and for decoding against a KV cache.

```python
import flashattn_cuda

o = flashattn_cuda.prefill(q, k, v, causal=True, scale=1/8)   # (B, H, N, 64) fp16
o = flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, scale=1/8)
```

## Numbers

Every number in this README, in [docs/benchmarks.md](docs/benchmarks.md) and in
[docs/profiles/analysis.md](docs/profiles/analysis.md) was measured on **one machine**:
RTX 3090 (sm_86, 82 SMs), driver 580.126.09, CUDA 13.0, PyTorch 2.11.0+cu130,
flash-attn 2.8.3. Nothing is carried over from an earlier environment.

Prefill, **B=8, H=12, causal**, median of three runs of 30 launches:

| N | v2 prefill | torch SDPA (flash) | flash-attn 2.8.3 | v1 fused (scaled to B·H) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1.238 ms · 10.40 TFLOP/s | 0.345 ms · 37.33 | 0.424 ms · 30.42 | 238.3 ms · 0.11 |
| 2048 | 4.575 ms · 11.26 TFLOP/s | 1.154 ms · 44.66 | 1.305 ms · 39.49 | 438.1 ms · 0.24 |
| 4096 | 17.644 ms · 11.68 TFLOP/s | 4.230 ms · 48.74 | 4.805 ms · 42.90 | 1092.4 ms · 0.38 |

Decode, single sequence, H=12, one token per call, against a **per-step eager SDPA loop**
(one `scaled_dot_product_attention` call per token over the whole cache):

| context | v2 decode | per-step eager SDPA loop |
| ---: | ---: | ---: |
| 128 | 0.0169 ms · 59,305 tok/s | 0.0222 ms · 45,003 tok/s |
| 512 | 0.0170 ms · 58,947 tok/s | 0.0303 ms · 32,992 tok/s |
| 1024 | 0.0173 ms · 57,899 tok/s | 0.0300 ms · 33,368 tok/s |
| 2048 | 0.0314 ms · 31,844 tok/s | 0.0302 ms · 33,066 tok/s |

Note the last row: **at context 2048 the eager loop is marginally faster** (0.0302 ms against
0.0314). The split kernel's advantage comes from the parallelism it extracts at small context,
where a single-row query otherwise leaves the card almost idle; by 2048 the eager loop has
enough work to fill the machine on its own and the split kernel's merge pass stops paying for
itself. That crossover is real and is left in rather than trimmed to a flattering range.

That baseline is a correct reference, not a tuned decode kernel: it does not split the KV
dimension, so it leaves unused exactly the parallelism this kernel exploits. The serious
baselines are flash-attn's dedicated decode path (`flash_attn_with_kvcache`) and FlashDecoding,
which do split KV and run on tensor cores. Measuring against those is deferred, so read the
decode numbers as "faster than the obvious PyTorch way to decode" and not as a claim about
state-of-the-art decode kernels.

![Prefill throughput, B=8, causal](docs/charts/prefill_b8_causal.svg)

### The honest gap

**v2 prefill is 2.9–3.7x slower than flash-attn** across N=1024–4096 at B=8 causal (4.2x
against `torch.sdpa`'s flash backend at N=4096), and **62x faster** than the v1 kernels it
replaces at N=4096 — 192x at N=1024, where v1's lack of batching hurts most. Small shapes go
the other way: at B=1, N=128, v2 takes 0.019 ms against SDPA's 0.056 and flash-attn's 0.095,
because at that size both libraries are dominated by fixed launch overhead that a simpler
kernel does not pay. `torch.sdpa`'s flash backend and the `flash-attn` package track each
other closely, as expected — SDPA dispatches to the same FlashAttention-2 kernels.

The gap is not a tuning gap. v2 sustains 10.4–11.7 TFLOP/s, which is **29–32% of this card's
36.2 TFLOP/s fp32 FMA peak** — a respectable fraction of the machine this kernel is using. The
problem is *which* machine it uses: every multiply happens on the CUDA cores, and the operands
arrive through shared memory at roughly 1.75 bytes per FMA while an SM sustains about 1.0. On
that model the design caps near 57% of fp32 peak before staging, softmax bookkeeping and
barriers are paid for. flash-attn is not winning by being 3x better at the same game — it runs
the matmuls on the tensor cores, whose fp16 peak is about 2x the fp32 pipe and which take
their operands from registers via `ldmatrix`/`mma` instead of the shared-memory path that
limits this kernel.

**What is measured and what is modeled.** The times above are measured, by a clock. The
TFLOP/s, GB/s and "% of peak" figures are *derived* from those times and the analytic FLOP and
byte counts — they assume the kernel moves exactly the bytes the model says. The
shared-memory-limited explanation of the gap is likewise a **model**: confirming which unit
actually saturates needs hardware performance counters, and the machine these numbers come
from denies access to them (`ERR_NVGPUCTRPERM` — it is a container, and the restriction is a
per-host driver property that cannot be lifted from inside).

**Counter-validated profiles are therefore pending**, tracked as open work against a
documentation-only `v2.0.1` cut at the `v2.0.0` commit — see
[docs/profiles/summary.md](docs/profiles/summary.md) for what will be captured and how. Until
then, treat every claim here about which hardware unit is saturated as a model that has not
been counter-validated. The reproducible split is set out in
[docs/profiles/analysis.md](docs/profiles/analysis.md).

So closing the gap means writing a WMMA/MMA kernel, not shaving instructions off this one. That
is deliberately out of scope for v2.0.0 and is the next planned step.

**Decode picks its split size per launch.** A decode call has only `batch x heads x splits`
blocks, so the split-K chunk decides whether the GPU is filled at all: with the chunk fixed at
512 keys, one sequence at context 1024 launched 24 blocks on an 82-SM card, with an access
pattern that was already one fully coalesced 128-byte request per cached row. `decode` now
chooses the largest power-of-two chunk in [128, 1024] for which `batch · heads · ceil(S/chunk)`
reaches two blocks per SM, falling back to the finest split when no chunk can —
largest-that-fits rather than finest-available, so the merge pass stays cheap. At B=1, H=12
that is worth **2.9–3.0x**:

| S | fixed Sk=512 | adaptive | speedup |
| ---: | ---: | ---: | ---: |
| 512 | 0.0514 ms · 30.7 GB/s | 0.0175 ms · 91.5 GB/s | 2.94x |
| 1024 | 0.0517 ms · 61.1 GB/s | 0.0175 ms · 182.5 GB/s | 2.95x |
| 2048 | 0.0909 ms · 69.5 GB/s | 0.0302 ms · 211.6 GB/s | 3.01x |

The chunk size is a scheduling choice, not a numerical one — the log-sum-exp merge makes the
result independent of it, and the suite asserts that across every chunk in the range. Full
analysis in [docs/profiles/analysis.md](docs/profiles/analysis.md).

## What is in v2

| | |
| :-- | :-- |
| shapes | `(B, H, N, 64)`, contiguous row-major, fp16 |
| accumulation | fp32 throughout (scores, softmax statistics, output) |
| prefill | one block per `(batch, head, query tile)`, Br=64 x Bc=32, online softmax |
| causal | key tiles above the diagonal are skipped, not masked |
| decode | split-K over the cached keys, log-sum-exp merge of the partials |
| ragged batches | `seq_lens` is per-sequence; chunks past a sequence's end drop out |
| verification | 90 GPU tests against an fp32 SDPA reference, tolerance 2e-3 |

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
python -m pytest -m gpu           # 90 GPU tests
```

The suite compares against `scaled_dot_product_attention` computed in fp32 on upcast inputs, so
the measured difference is the kernel's error and not the reference's. It sweeps
B in {1,4} x H in {8,12,14} x N in {128,512,1024,2048,4096} x causal, plus sequence lengths that
are not tile multiples, decode against the causal prefill output, ragged decode batches, and
cache lengths straddling the split boundary.

## Benchmark and profile

```bash
python bench/run_bench.py       # timings and charts   -> docs/benchmarks.md
python bench/run_analysis.py    # occupancy + roofline -> docs/profiles/analysis.md
nsys profile -t cuda -o docs/profiles/prefill python bench/profile_workload.py prefill
bash bench/profile_ncu.sh       # counter capture      -> docs/profiles/summary.md
```

Full tables, chart sources and the machine description are in
[docs/benchmarks.md](docs/benchmarks.md); occupancy, the analytical roofline and the
timeline notes are in [docs/profiles/analysis.md](docs/profiles/analysis.md).
`bench/profile_ncu.sh` needs a card whose driver allows performance counters, and refuses
to run with instructions where it does not.

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
