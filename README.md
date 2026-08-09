# Flash Attention CUDA 🚀

[![CI](https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900.svg)](https://docs.nvidia.com/cuda/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu130-EE4C2C.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A **Flash Attention implementation written completely from scratch in CUDA**, packaged as an installable PyTorch extension.

This project speeds up the Attention operation used inside Large Language Models (LLMs) like ChatGPT, Llama and Gemini, by using the GPU efficiently.

Instead of using PyTorch's built-in attention, this project performs the calculations directly on the GPU using hand-written CUDA kernels.

**Repository:** https://github.com/shrvan30/flash-attention-cuda

### Supports

- Batched Multi-Head Attention
- Causal Attention
- KV Cache Decoding
- FP16 Input + FP32 Accumulation
- Separate Prefill and Decode CUDA Kernels
- Split-K Decode
- Ragged (variable-length) sequences

---

## Table of Contents

- [Quick Example](#quick-example)
- [Why did I build this?](#why-did-i-build-this)
- [What is Attention?](#what-is-attention)
- [The Problem](#the-problem)
- [What is Flash Attention?](#what-is-flash-attention)
- [What did I implement?](#what-did-i-implement)
- [Features](#features)
- [GPU Optimizations Used](#gpu-optimizations-used)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Installation](#installation)
- [Testing](#testing)
- [Benchmarking and Profiling](#benchmarking-and-profiling)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Continuous Integration](#continuous-integration)
- [What I Learned](#what-i-learned)
- [References](#references)
- [License](#license)

---

## Quick Example

```python
import torch
import flashattn_cuda

# ---- Prompt processing (prefill) ----
# q, k, v : (batch, heads, seq_len, head_dim) in FP16 on CUDA
out = flashattn_cuda.prefill(
    q,
    k,
    v,
    causal=True,
    scale=1 / 8,
)

# ---- Token generation (decode) ----
# q        : (batch, heads, 1, head_dim)
# k_cache  : (batch, heads, max_len, head_dim)
# v_cache  : (batch, heads, max_len, head_dim)
# seq_lens : (batch,) int32 — current valid length per sequence
out = flashattn_cuda.decode(
    q,
    k_cache,
    v_cache,
    seq_lens,
    scale=1 / 8,
)
```

---

## Why did I build this?

Large Language Models spend most of their execution time inside the Attention layer.

For long sentences, attention becomes slow because:

- It performs billions of multiplications.
- It repeatedly moves data between GPU memory and processors.
- Memory access becomes the bottleneck instead of computation.

The goal of this project was to **learn how FlashAttention works internally and implement it from scratch**, rather than calling a library.

---

## What is Attention?

Imagine 5 students sitting in a classroom.

When Student 5 answers a question, he first looks at all previous students to understand the discussion.

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

The final answer is produced by combining information from all previous students according to these percentages.

This process is called **Attention**.

LLMs perform this operation for every word in the sentence.

---

## The Problem

Suppose we have:

```
4000 words
```

Each word compares itself with every previous word.

That means:

```
4000 × 4000
=
16 million comparisons
```

Doing this repeatedly becomes expensive.

Even worse, the GPU wastes time **moving data between memory** instead of doing calculations.

---

## What is Flash Attention?

Instead of loading the entire sentence into GPU memory, Flash Attention divides it into small blocks.

Instead of:

```
4000 words
```

it processes:

```
64 words
   ↓
next 64
   ↓
next 64
```

Only one small block is kept inside the GPU's fast shared memory.

```
Less memory movement
        ↓
More GPU utilization
        ↓
  Faster execution
```

---

## What did I implement?

I wrote two CUDA kernels.

### 1. Prefill Kernel

Used when the entire prompt is already available.

```
User: Explain Newton's Laws.
```

The model already knows the whole sentence, so every word attends to all previous words.

### 2. Decode Kernel

Used while the model is generating new words.

```
The cat sat on the ______
```

The model predicts one new word at a time. Instead of recomputing attention for the entire sentence, it stores previous Keys and Values inside a **KV cache**. Every new token only compares against the cache.

This makes generation much faster.

---

## Features

| Feature | Status |
|---|---|
| Batched Multi-Head Attention | ✔ |
| Causal Attention | ✔ |
| KV Cache Decode | ✔ |
| FP16 Inputs | ✔ |
| FP32 Accumulation | ✔ |
| Shared Memory Tiling | ✔ |
| Online Softmax | ✔ |
| Split-K Decode | ✔ |
| Ragged Sequence Support | ✔ |
| PyTorch CUDA Extension (`pip install -e .`) | ✔ |
| 90 GPU Correctness Tests | ✔ |
| Tensor Cores / WMMA / MMA | ✘ (planned) |
| Backward Pass / Training | ✘ (planned) |

---

## GPU Optimizations Used

### Shared Memory

Frequently used data is copied into the GPU's fast shared memory. Instead of reading from slow global memory repeatedly, threads reuse the same shared data.

### Tiling

The attention matrix is divided into small blocks.

```
4000 × 4000
     ↓
 64 × 32 blocks
```

Each CUDA block processes one tile.

### Online Softmax

Normally softmax needs multiple passes. Flash Attention computes softmax **while** processing tiles, using a running maximum and a running sum. This reduces memory usage.

### Causal Attention

Words should not see future words.

```
I love ______
```

The model should not look at `pizza` before predicting it. Future positions are skipped automatically.

### FP16 Inputs

The input tensors are stored using FP16. This saves memory and bandwidth.

### FP32 Accumulation

Although inputs are FP16, all mathematical accumulation happens in FP32. This improves numerical accuracy.

### Split-K Decode

During decoding, multiple CUDA blocks process different parts of the KV cache. Their partial results are merged later. This increases GPU utilization when the batch is small.

---

## Performance

Measured on an NVIDIA RTX 3090 (SM 8.6).

| Path | Result |
|---|---|
| **Prefill** | Up to **62× faster** than my previous (v1) implementation |
| **Decode** | Around **3× faster** than a simple PyTorch decode loop for small contexts |

Full tables, methodology and profiler traces are in [`docs/benchmarks.md`](docs/benchmarks.md) and [`docs/profiles/analysis.md`](docs/profiles/analysis.md).

---

## Project Structure

```
flash-attention-cuda
│
├── csrc/                  # CUDA kernels + bindings
│   ├── prefill.cu         # Prefill (full prompt) kernel
│   ├── decode.cu          # Decode (KV cache) kernel
│   ├── bindings.cpp       # PyTorch extension bindings
│   └── legacy/            # v1 kernels, kept for comparison
│
├── bench/                 # Benchmark + profiling scripts
│   ├── run_bench.py
│   ├── run_analysis.py
│   └── profile_workload.py
│
├── tests/                 # 90 GPU correctness tests
│
├── docs/                  # Performance reports
│   ├── benchmarks.md
│   ├── v1.md
│   └── profiles/
│       ├── analysis.md
│       └── summary.md
│
├── .github/workflows/ci.yml
├── LICENSE
├── setup.py
└── README.md
```

---

## Development Environment

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 3090 (SM 8.6) |
| CUDA | 13.0 |
| PyTorch | 2.11.0 + cu130 |
| OS | Ubuntu (WSL) |
| Profilers | Nsight Systems, Nsight Compute |
| Language | CUDA C++, C++, Python |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shrvan30/flash-attention-cuda.git
cd flash-attention-cuda
```

Install as an editable PyTorch extension:

```bash
pip install -e .
```

> **Requirements:** An NVIDIA GPU with compute capability 7.0+, a matching CUDA toolkit, and a PyTorch build compiled against the same CUDA version.

---

## Testing

Install pytest:

```bash
pip install pytest
```

Run the GPU correctness tests:

```bash
python -m pytest -m gpu
```

The project contains **90 GPU correctness tests** covering:

- Prefill
- Decode
- Causal Attention
- Ragged Sequences
- Multiple Batch Sizes
- Multiple Sequence Lengths

All kernels are validated against a PyTorch reference implementation.

---

## Benchmarking and Profiling

Generate benchmark tables:

```bash
python bench/run_bench.py
```

Generate profiling analysis:

```bash
python bench/run_analysis.py
```

Profile with Nsight Systems:

```bash
nsys profile -t cuda \
  -o docs/profiles/prefill \
  python bench/profile_workload.py prefill
```

```bash
nsys profile -t cuda \
  -o docs/profiles/decode \
  python bench/profile_workload.py decode
```

---

## Documentation

For detailed technical explanations, benchmarking and profiling reports, see:

| Document | Contents |
|---|---|
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmark tables and methodology |
| [`docs/profiles/analysis.md`](docs/profiles/analysis.md) | Nsight Compute kernel-level analysis |
| [`docs/profiles/summary.md`](docs/profiles/summary.md) | High-level profiling summary |
| [`docs/v1.md`](docs/v1.md) | The original v1 implementation and what changed |

---

## Limitations

The current version **uses**:

✔ CUDA cores
✔ Shared memory
✔ Online softmax
✔ Batched attention
✔ Multi-head attention
✔ KV Cache
✔ Causal masking

The current version **does not use**:

- Tensor Cores
- WMMA
- MMA Instructions

Because of this, it is still slower than NVIDIA's official FlashAttention implementation.

---

## Future Improvements

- Tensor Core (WMMA) implementation
- MMA instructions
- Higher occupancy
- Better memory throughput
- Support for more head dimensions
- Backward pass
- Training support

---

## Continuous Integration

Every commit automatically runs:

- Build
- Unit tests
- GPU test suite (when a GPU runner is available)

**Workflow file:** [`.github/workflows/ci.yml`](https://github.com/shrvan30/flash-attention-cuda/blob/main/.github/workflows/ci.yml)

**Workflow runs:** https://github.com/shrvan30/flash-attention-cuda/actions/workflows/ci.yml

---

## What I Learned

While building this project I learned:

- CUDA Programming
- GPU Architecture
- Memory Hierarchy
- Shared Memory
- Warp Execution
- Occupancy
- Memory Coalescing
- Online Softmax
- The FlashAttention Algorithm
- PyTorch CUDA Extensions
- GPU Profiling using Nsight

---

## References

| Resource | Link |
|---|---|
| FlashAttention Paper | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 Paper | https://arxiv.org/abs/2307.08691 |
| CUDA Programming Guide | https://docs.nvidia.com/cuda/ |
| CUDA C++ Best Practices Guide | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ |
| PyTorch CUDA Extension Tutorial | https://pytorch.org/tutorials/advanced/cpp_extension.html |
| Nsight Systems | https://developer.nvidia.com/nsight-systems |
| Nsight Compute | https://developer.nvidia.com/nsight-compute |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for full details.

---

## In One Line

> "I built Flash Attention completely from scratch in CUDA to understand how modern LLMs speed up attention using shared memory, tiling, online softmax, and GPU optimizations instead of relying on PyTorch's built-in implementation."
