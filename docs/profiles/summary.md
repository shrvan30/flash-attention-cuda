# Kernel profiles

Traces in this directory were collected with Nsight Systems on the reference machine:

| | |
| :-- | :-- |
| GPU | NVIDIA GeForce RTX 3090 (Ampere, sm_86, 24 GB, 82 SMs) |
| Driver | 595.71.05 |
| CUDA toolkit (nvcc) | 13.2 |
| PyTorch | 2.13.0+cu130 |

Reproduce with:

```bash
nsys profile -t cuda -o docs/profiles/prefill python bench/profile_workload.py prefill
nsys profile -t cuda -o docs/profiles/decode  python bench/profile_workload.py decode
nsys stats --report cuda_gpu_kern_sum docs/profiles/prefill.nsys-rep
```

## Scope of these numbers

> **Hardware performance counters were unavailable on the machine these profiles were taken
> on.** The NVIDIA driver was configured with `RmProfilingAdminOnly=1`, so every Nsight Compute
> run — including one asking only for static sections, and including one profiling a bare
> `torch.mm` — fails with `ERR_NVGPUCTRPERM`. That parameter can only be changed on the host,
> which is not reachable from this container.
>
> Consequently the counter-derived figures (SM speed-of-light %, memory speed-of-light %,
> *achieved* occupancy, measured DRAM throughput, shared-memory bank-conflict counts) are **not
> reported here rather than estimated**, and there are no `.ncu-rep` files. What is reported
> instead is measured kernel wall-clock from the Nsight Systems traces, achieved throughput
> derived from it, and *theoretical* occupancy computed from the register and shared-memory
> footprints that `ptxas` reports. Every one of those is a real measurement or an exact
> calculation; none is a stand-in for a counter.

Static footprints (`nvcc -arch=sm_86 -Xptxas -v`):

| kernel | registers/thread | shared/block | threads/block | spills |
| :-- | ---: | ---: | ---: | ---: |
| `prefill_kernel<causal>` | 103 | 41,472 B | 128 | 0 |
| `decode_split_kernel` | 37 | 1,056 B | 128 | 0 |
| `decode_merge_kernel` | 38 | 0 B | 64 | 0 |

## Prefill — N=2048, causal, B=4, H=12

From `prefill.nsys-rep` (25 launches):

| | |
| :-- | :-- |
| mean kernel time | 1.9516 ms |
| median / min / max | 1.9461 / 1.9274 / 1.9894 ms |
| standard deviation | 17.3 µs (0.9%) |
| useful work | 25.77 GFLOP (`4·B·H·N²·d`, halved for causal) |
| achieved | **13.20 TFLOP/s** |
| fp32 FMA peak (82 SM × 128 lanes × 2 × 1.695 GHz) | 35.6 TFLOP/s |
| fraction of fp32 peak | **37%** |
| theoretical occupancy | **16.7%** (8 of 48 warps per SM) |

Occupancy is shared-memory limited, not register limited: 41,472 B per block against the
100 KB an SM can hand out allows 2 blocks, while 103 registers × 128 threads would allow 4.
The grid is 32 query tiles × 12 heads × 4 batch = 1,536 blocks against 164 resident, i.e. about
9.4 waves, so the tail effect is negligible and the 0.9% run-to-run spread confirms it.

**Top bottleneck: shared-memory bandwidth in the two inner matmuls.** The QK loop reads two
16-byte vectors per 16 FMAs and the PV loop three per 32 FMAs — about 1.75 bytes of shared
traffic per FMA. An SM sustains 128 B/cycle of shared memory against 128 FMA/cycle, i.e. 1.0
byte per FMA, so this blocking cannot exceed roughly 57% of FMA peak no matter how well the rest
behaves; staging, softmax bookkeeping and three barriers per tile account for the gap between
that ceiling and the observed 37%. DRAM is not the constraint: including causal tile skipping
the kernel touches ~440 MB of Q/K/V/O per launch, and even assuming zero L2 reuse that is
225 GB/s against 936 GB/s of peak bandwidth.

The way to break the ceiling is not more tuning of this shape — it is tensor cores, which
deliver operands through a different path (`ldmatrix` into MMA fragments) and would remove the
per-FMA shared traffic that sets the limit. That is the documented gap against `flash-attn` in
[../benchmarks.md](../benchmarks.md), and it is Phase 1's explicit stretch goal.

## Decode — S=1024, B=1, H=12

From `decode.nsys-rep` (25 launches of each kernel):

| kernel | mean | share of the call |
| :-- | ---: | ---: |
| `decode_split_kernel` | 37.58 µs | 93.7% |
| `decode_merge_kernel` | 1.80 µs | 4.5% |
| total | 39.38 µs | |

| | |
| :-- | :-- |
| KV bytes read | 3.15 MB (`2 · H · S · 64 · 2`) |
| achieved bandwidth | **83.7 GB/s** |
| peak bandwidth | 936 GB/s |
| fraction of peak | **8.9%** |
| blocks launched | 24 (`ceil(1024/512)` splits × 12 heads × 1 batch) |
| SMs on the device | 82 |

**Top bottleneck: not enough blocks to fill the machine.** Decode is a bandwidth-bound problem —
two FLOPs per byte of cache read — but at this shape it never gets the chance to be bandwidth
bound: 24 blocks means at most 24 of 82 SMs have any work at all, and each of those runs a
single 4-warp block, so there are nowhere near enough concurrent memory requests in flight to
saturate HBM. The per-SM access pattern is already ideal (one warp reads one cached row as a
single 128-byte coalesced request), which is why the fix is not in the inner loop.

The lever is `Sk`, fixed at 512 for this release. Dropping it to 128 would produce 96 blocks at
this shape — more than the SM count — at the cost of 4× more partials for the merge pass, which
currently takes 4.5% of the call. The same effect appears for free as batch size grows: at B=8
the identical kernel launches 192 blocks. Tuning `Sk` (and the paged KV cache that would change
the access pattern anyway) is deliberately left outside v2.0.0.

`decode_merge_kernel` at 1.80 µs for 24 blocks of 64 threads is essentially launch latency; it
reads 24 × 2 partials and writes 768 halves. It only starts to matter if `Sk` shrinks a lot.
