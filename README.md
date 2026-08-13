# Flash Attention CUDA

[![CI](https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900.svg)](https://docs.nvidia.com/cuda/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu130-EE4C2C.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

> **What did I build?** A from-scratch CUDA FlashAttention kernel — shared-memory
> tiling, online softmax, fp16 inputs with fp32 accumulation, separate prefill and
> split-K decode paths — packaged as a pip-installable PyTorch extension and plugged
> into a real inference engine ([FlashStack](https://github.com/shrvan30/flashstack))
> to measure what a hand-written kernel is actually worth end to end.

---

## Table of Contents

- [What can it actually do, today?](#what-can-it-actually-do-today)
- [Motivation](#motivation)
- [What is Attention? (explained simply)](#what-is-attention-explained-simply)
- [The Problem](#the-problem)
- [What is Flash Attention?](#what-is-flash-attention)
- [What did I implement?](#what-did-i-implement)
- [Kernel design](#kernel-design)
- [Performance](#performance)
- [Engineering decisions](#engineering-decisions)
- [Challenges and debugging](#challenges-and-debugging)
- [Limitations](#limitations)
- [Project structure](#project-structure)
- [Environment (every published number comes from here)](#environment-every-published-number-comes-from-here)
- [Installation](#installation)
- [Testing](#testing)
- [Reproduce every number](#reproduce-every-number)
- [Documentation](#documentation)
- [Tests / CI](#tests--ci)
- [Future work](#future-work)
- [References](#references)
- [License](#license)

---

## What can it actually do, today?

No buzzwords — the concrete capability list, and what is deliberately missing.

**You can:**

- Compute attention for any model whose **head size is 64** — which covers **GPT-2**
  and **Qwen2.5-0.5B-Instruct**, the two models this kernel actually runs inside
  (see [FlashStack](https://github.com/shrvan30/flashstack)).
- **Prompt phase (prefill):** batched, multi-head, causal or full attention, fp16
  inputs, tested to sequence length 4096.
- **Generation phase (decode):** one new token attending over a KV cache, with a
  different current length per sequence in the batch, using a split-K kernel that
  **picks its own chunk size at launch** so even batch-size-1 fills the GPU.
- Call it from Python with two functions — `prefill()` and `decode()` — after a
  single `pip install -e .`.

**Guarantees, not vibes:** outputs match a full-fp32 PyTorch reference within
**2.44e-4**, and decoding token *t* from the cache equals row *t* of prefill
(99 tests enforce both, 90 of them on GPU).

**It cannot (yet):** head sizes other than 64, training/backward, tensor cores —
which is exactly why it is **~3–4x slower than NVIDIA's flash-attn**, with
[`docs/profiles/analysis.md`](docs/profiles/analysis.md) explaining why tensor
cores are the only door past the current ceiling.

---

## Motivation

Large Language Models spend most of their execution time inside the Attention layer.

For long sequences, attention becomes slow because:

- It performs billions of multiplications.
- It repeatedly moves data between GPU memory and processors.
- Memory access becomes the bottleneck instead of computation.

The goal of this project was to **learn how FlashAttention works internally by
implementing it from scratch**, then to find out — with a profiler, in a real
serving stack — how much the kernel actually matters. (Spoiler from the companion
repo: less than the layer above it. That finding is the point.)

---

## What is Attention? (explained simply)

Imagine 5 students sitting in a classroom.

When Student 5 answers a question, he first looks at all previous students to
understand the discussion.

```
S1
S2
S3
S4
S5  <-- currently speaking
```

Student 5 pays different amounts of attention to each previous student.

Maybe:

```
S1 -> 10%
S2 -> 20%
S3 -> 50%
S4 -> 20%
```

The final answer combines information from all previous students according to
these percentages. This process is called **Attention**, and LLMs perform it for
every word in the sentence.

---

## The Problem

Suppose we have:

```
4000 words
```

Each word compares itself with every previous word:

```
4000 x 4000 = 16 million comparisons
```

The math is cheap. The killer is that the naive implementation writes that whole
16-million-entry score table to slow GPU memory, reads it back for softmax, writes
it again, reads it again. The GPU spends its time **moving data**, not computing.

---

## What is Flash Attention?

Instead of materializing the whole table, Flash Attention processes the sentence
in small blocks:

```
64 words -> next 64 -> next 64 -> ...
```

Only one small block lives in the GPU's fast shared memory at a time, and the
softmax is computed *while* streaming through blocks (a running maximum and a
running sum per row — "online softmax").

```
Less memory movement -> more GPU utilization -> faster execution
```

Memory drops from O(N^2) to O(N): the big table is never written anywhere.

---

## What did I implement?

Two CUDA kernels, because inference has two very different phases.

### 1. Prefill kernel

Used when the entire prompt is available at once:

```
User: Explain Newton's Laws.
```

Every token attends to all previous tokens — a big, parallel, GPU-friendly job.

### 2. Decode kernel

Used while generating, one token at a time:

```
The cat sat on the ______
```

Previous Keys and Values are stored in a **KV cache**; each new token only attends
over the cache. One query row versus thousands of cached rows has almost no natural
parallelism — so this kernel splits the cache into chunks (**split-K**), computes a
partial result per chunk, and merges the partials exactly. The chunk size is chosen
**at launch time** to keep about two blocks per SM in flight, which is worth ~3x
over a fixed chunk at batch size 1.

---

## Kernel design

| Technique | What it does here |
|---|---|
| Shared-memory tiling | K/V staged through fast on-chip memory in 64x32 tiles; the score matrix never touches HBM |
| Online softmax | Running max `m` and running sum `l` per row, in fp32 registers; output rescaled when the max moves |
| Causal tile skipping | Tiles entirely above the diagonal are skipped by loop bounds (verified bit-identical), ~1.8–1.9x at N=4096 |
| fp16 in, fp32 accumulate | Half the bandwidth of fp32 inputs, near-fp32 accuracy on the sums |
| Split-K decode + adaptive chunk | Parallelism manufactured from the KV dimension; `choose_decode_split` targets ~2 blocks/SM |
| Two-pass merge, no atomics | Partial `(m, l, acc)` merged by a small second kernel — deterministic and exact |

---

## Performance

Measured on **environment E1** — RTX 3090 (sm_86), driver 580.126.09, CUDA 13.0,
torch 2.11.0+cu130. Every number in this repository comes from that one machine.

| Comparison | Result |
|---|---|
| **vs official flash-attn 2.8.3 (prefill)** | **~3–4x slower** depending on shape (e.g. 0.698 ms vs 0.211 ms at B=1, H=12, N=2048) — no tensor cores; sustained ~31–34% of the fp32 FMA peak, with a modeled ceiling near ~57% from shared-memory bandwidth |
| vs torch SDPA, flash backend (prefill) | same picture — SDPA and flash-attn are within noise of each other |
| **vs a per-step eager SDPA loop (decode)** | **1.3–1.8x faster** up to context 1024; **roughly equal at 2048** (the eager loop is marginally faster there — published, not hidden). This baseline is deliberately labelled a low bar; the serious decode baselines (flash-attn's `flash_attn_with_kvcache`, FlashDecoding) are deferred and named in the docs |
| vs my v1 kernels (fp32, single-head) | up to 62x faster — a progress marker only, not a serious baseline |

Full tables and methodology: [`docs/benchmarks.md`](docs/benchmarks.md).
Bottleneck analysis (measured vs modeled, explicitly separated):
[`docs/profiles/analysis.md`](docs/profiles/analysis.md).

---

## Engineering decisions

The choices an interviewer should ask about, with the one-line reasons:

- **fp32 accumulation under fp16 inputs** — thousands-term sums in fp16 drift;
  fp32 keeps the error at 2.44e-4 while HBM traffic stays fp16.
- **Head size fixed at 64** — sizes the whole shared-memory/register budget, and
  matches the two target models exactly. d=128 is the named next step, not an
  accidental omission.
- **Separate decode kernel** — decode's shape (1 query row vs S cached rows) makes
  prefill's parallelization collapse; pretending one kernel fits both phases is how
  tutorials stay slow.
- **Adaptive split size, chosen at launch** — a fixed chunk starved the GPU
  (24 blocks on 82 SMs at batch 1); the fix is scheduling, not inner-loop code, and
  a test proves every legal chunk size produces identical output.
- **Two-pass merge instead of atomics** — the softmax merge needs `(m, l)`
  rescaling, which atomics cannot express deterministically.
- **Baselines labelled, always** — the decode reference is named as "the obvious
  PyTorch way", and beating it is called a low bar in the docs themselves.

---

## Challenges and debugging

- **The starved decode.** First decode version ran at 9% of peak bandwidth at
  batch 1. The kernel code was fine — the launch geometry put 24 blocks on an
  82-SM GPU. Lesson: a slow kernel and a starved GPU look identical from outside;
  the trace tells them apart.
- **Profile the artifact you ship.** Standalone `nvcc -Xptxas -v` reported 103
  registers/thread; the real build uses 121, because `-lineinfo` and torch's flags
  change allocation. All published resource numbers come from the shipped build via
  `cudaFuncGetAttributes`.
- **Counters denied.** Cloud containers inherit the host driver's profiling
  restriction (`ERR_NVGPUCTRPERM`), so Nsight Compute could not run. Instead of
  estimating, the analysis splits every figure into **measured** (clocks, traces)
  and **modeled** (rooflines, theoretical occupancy) and says which is which —
  see [`docs/profiles/summary.md`](docs/profiles/summary.md) for the tracked plan
  to counter-validate.

---

## Limitations

Stated up front, because hiding them would cost more than they do:

- Head dimension is fixed at **64**.
- **No tensor cores** (WMMA/MMA) — the single biggest reason for the gap to
  flash-attn.
- **No backward pass** — inference only.
- Decode's advantage over even the eager-loop baseline **disappears at context
  2048**; state-of-the-art decode kernels have not been benchmarked against yet.
- Which hardware unit saturates is currently a **model, not a counter-validated
  measurement** ([`docs/profiles/summary.md`](docs/profiles/summary.md) tracks
  closing this as v2.0.1).

---

## Project structure

```
flash-attention-cuda
├── csrc/                  # CUDA kernels + bindings
│   ├── prefill.cu         # Prefill (full prompt) kernel
│   ├── decode.cu          # Decode (KV cache, split-K, adaptive chunk)
│   ├── bindings.cpp       # PyTorch extension bindings + input validation
│   └── legacy/            # v1 kernels, kept for comparison
├── bench/                 # run_bench.py, run_analysis.py, profile_workload.py, profile_ncu.sh
├── tests/                 # 99 tests (90 GPU-marked)
├── docs/                  # benchmarks.md, v1.md, profiles/{analysis.md, summary.md}
├── .github/workflows/ci.yml
├── setup.py
└── README.md
```

---

## Environment (every published number comes from here)

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 3090 (sm_86, 24 GB, 82 SMs) — rented cloud Linux box ("E1") |
| Driver | 580.126.09 |
| CUDA toolkit | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12 |
| Profilers | Nsight Systems (Nsight Compute pending — see [`docs/profiles/summary.md`](docs/profiles/summary.md)) |

---

## Installation

```bash
git clone https://github.com/shrvan30/flash-attention-cuda.git
cd flash-attention-cuda
pip install -e .
```

> **Tested on:** RTX 3090 (sm_86, Ampere) with CUDA 13.0 and torch 2.11+cu130.
> Other architectures should compile via `TORCH_CUDA_ARCH_LIST=<your arch>` but
> are unverified. The PyTorch build's CUDA major must match `nvcc`'s.

### Quick example

```python
import flashattn_cuda

# Prompt phase: q, k, v are (batch, heads, seq_len, 64), fp16, CUDA
out = flashattn_cuda.prefill(q, k, v, causal=True, scale=1 / 8)

# Generation phase: q is (batch, heads, 1, 64); caches are (batch, heads, max_len, 64)
out = flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, scale=1 / 8)
```

---

## Testing

```bash
pip install pytest
python -m pytest            # CPU-runnable checks
python -m pytest -m gpu     # 90 GPU correctness tests
```

Coverage: prefill and decode vs a full-fp32 SDPA reference across batch sizes,
head counts, lengths 128–4096, causal and full; ragged (mixed-length) decode;
decode-vs-prefill consistency; output invariance across every legal split size.

---

## Reproduce every number

```bash
python bench/run_bench.py        # regenerates docs/benchmarks.md tables + charts
python bench/run_analysis.py     # regenerates docs/profiles/analysis.md

# Nsight Systems traces (analysis reads these back):
nsys profile -t cuda -o docs/profiles/prefill  python bench/profile_workload.py prefill
nsys profile -t cuda -o docs/profiles/decode   python bench/profile_workload.py decode
```

The generated `.nsys-rep` captures are **not committed** — profiler traces embed
launch metadata, so they ship as release assets instead of living in git. On a
counter-capable machine, `bash bench/profile_ncu.sh` produces the Nsight Compute
profiles described in [`docs/profiles/summary.md`](docs/profiles/summary.md).

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmark tables, methodology, and the explicit baseline definitions |
| [`docs/profiles/analysis.md`](docs/profiles/analysis.md) | Kernel-level performance analysis — nsys traces + CUDA events (**measured**) and roofline/occupancy (**modeled**), labelled as such |
| [`docs/profiles/summary.md`](docs/profiles/summary.md) | The Nsight Compute plan: what counters will add, why they are absent, how to produce them |
| [`docs/v1.md`](docs/v1.md) | The original v1 progression (CPU → naive → tiled → fused), kept for history |

---

## Tests / CI

Every commit runs:

- ruff / clang-format lint
- a full **CUDA compile check** of the extension inside an `nvidia/cuda` devel
  container — no GPU device needed to prove the kernels build

GPU correctness tests (`pytest -m gpu`) run on real hardware **before every tag**,
not in CI: there is no GPU runner, and this README won't pretend otherwise.

---

## Future work

- Tensor cores (WMMA) for prefill — the measured ceiling says this is the door
- Head dimension 128 (unlocks larger models in FlashStack)
- Kernel-side GQA (stop paying the host-side KV repeat)
- Benchmarks against flash-attn's decode path and FlashDecoding
- Counter-validated profiles (tracked as v2.0.1)
- Backward pass / training

---

## References

| Resource | Link |
|---|---|
| FlashAttention paper | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 paper | https://arxiv.org/abs/2307.08691 |
| CUDA Programming Guide | https://docs.nvidia.com/cuda/ |
| CUDA C++ Best Practices | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ |
| PyTorch CUDA extension tutorial | https://pytorch.org/tutorials/advanced/cpp_extension.html |
| Nsight Systems | https://developer.nvidia.com/nsight-systems |
| Nsight Compute | https://developer.nvidia.com/nsight-compute |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |

---

## License

MIT — see [LICENSE](LICENSE).

---

## In one line

> "I built FlashAttention from scratch in CUDA — then plugged it into a real
> serving stack and measured, with a profiler, exactly how much a kernel is worth
> end to end."
