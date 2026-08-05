"""On-box performance analysis: occupancy, roofline and the decode split study.

Writes docs/profiles/analysis.md. Uses CUDA-event timings and the CUDA occupancy
API only — this machine has no access to GPU performance counters, so nothing
here comes from a profiler. The counter-based report is produced separately by
bench/profile_ncu.sh on a counter-capable card.

    python bench/run_analysis.py
"""

import csv
import math
import os
import subprocess

import torch

import flashattn_cuda

HEAD_DIM = flashattn_cuda.head_dim
SCALE = 1.0 / math.sqrt(HEAD_DIM)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "docs", "profiles", "analysis.md")

# GA102 / RTX 3090 peaks.
DRAM_PEAK_GBS = 936.0


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return sorted(samples)[1]


def fp32_fma_peak_tflops(props):
    """82 SMs x 128 fp32 lanes x 2 flops/FMA x boost clock."""
    return props.multi_processor_count * 128 * 2 * (props.clock_rate * 1e3) / 1e12


def nvidia_smi(query):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], text=True
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def prefill_roofline(props, shapes):
    rows = []
    compute_peak = fp32_fma_peak_tflops(props)
    for batch, heads, seq, causal in shapes:
        q, k, v = (
            torch.randn(batch, heads, seq, HEAD_DIM, device="cuda", dtype=torch.float16)
            for _ in range(3)
        )
        ms = timeit(lambda: flashattn_cuda.prefill(q, k, v, causal, SCALE))

        flops = 4.0 * batch * heads * seq * seq * HEAD_DIM
        if causal:
            flops *= 0.5
        # Minimum traffic: three reads and one write of a (B, H, N, 64) fp16
        # tensor. K/V are re-read per query tile, but those re-reads are what the
        # L2 exists to absorb, so this is the roofline denominator.
        move_bytes = 4.0 * batch * heads * seq * HEAD_DIM * 2
        tflops = flops / (ms * 1e-3) / 1e12
        gbs = move_bytes / (ms * 1e-3) / 1e9
        rows.append(
            {
                "shape": f"B={batch}, H={heads}, N={seq}, {'causal' if causal else 'full'}",
                "ms": ms,
                "tflops": tflops,
                "pct_compute": 100.0 * tflops / compute_peak,
                "gbs": gbs,
                "pct_dram": 100.0 * gbs / DRAM_PEAK_GBS,
            }
        )
    return rows, compute_peak


def decode_roofline(props, shapes, split_size=0):
    """split_size = 0 uses the adaptive choice; a positive value forces it."""
    rows = []
    compute_peak = fp32_fma_peak_tflops(props)
    target = 2 * props.multi_processor_count
    for batch, heads, seq in shapes:
        q = torch.randn(batch, heads, 1, HEAD_DIM, device="cuda", dtype=torch.float16)
        k_cache = torch.randn(
            batch, heads, seq, HEAD_DIM, device="cuda", dtype=torch.float16
        )
        v_cache = torch.randn_like(k_cache)
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")

        if split_size:
            chunk = split_size
            fn = lambda: flashattn_cuda._decode_with_split(  # noqa: E731
                q, k_cache, v_cache, seq_lens, SCALE, split_size
            )
        else:
            chunk = flashattn_cuda.choose_decode_split(seq, batch, heads, target)
            fn = lambda: flashattn_cuda.decode(  # noqa: E731
                q, k_cache, v_cache, seq_lens, SCALE
            )
        ms = timeit(fn)

        splits = -(-seq // chunk)
        blocks = batch * heads * splits
        # KV cache read once, plus the split workspace written and read back.
        cache_bytes = 2.0 * batch * heads * seq * HEAD_DIM * 2
        workspace_bytes = 2.0 * batch * heads * splits * (HEAD_DIM + 2) * 4
        flops = 4.0 * batch * heads * seq * HEAD_DIM
        rows.append(
            {
                "shape": f"B={batch}, H={heads}, S={seq}",
                "chunk": chunk,
                "blocks": blocks,
                "ms": ms,
                "gbs": (cache_bytes + workspace_bytes) / (ms * 1e-3) / 1e9,
                "pct_dram": 100.0
                * ((cache_bytes + workspace_bytes) / (ms * 1e-3) / 1e9)
                / DRAM_PEAK_GBS,
                "tflops": flops / (ms * 1e-3) / 1e12,
                "pct_compute": 100.0 * (flops / (ms * 1e-3) / 1e12) / compute_peak,
            }
        )
    return rows


def nsys_kernel_stats(name):
    """Per-kernel timings read back out of an nsys trace in docs/profiles/.

    Read from the trace rather than restated in prose: a hand-written timeline
    figure survives the environment it was measured on, and then silently
    describes a machine that is no longer the one being published.

    Returns a list of dicts, slowest first, or None when the trace is missing or
    nsys cannot read it.
    """
    report = os.path.join(REPO_ROOT, "docs", "profiles", f"{name}.nsys-rep")
    if not os.path.exists(report):
        return None
    try:
        raw = subprocess.check_output(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                report,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    rows = []
    for line in raw.splitlines():
        parts = next(csv.reader([line]), [])
        if len(parts) < 9 or parts[0] == "Time (%)":
            continue
        try:
            rows.append(
                {
                    "pct": float(parts[0]),
                    "total_ns": float(parts[1]),
                    "instances": int(parts[2]),
                    "avg_ns": float(parts[3]),
                    "med_ns": float(parts[4]),
                    "min_ns": float(parts[5]),
                    "max_ns": float(parts[6]),
                    "stddev_ns": float(parts[7]),
                    "name": short_kernel_name(parts[8]),
                }
            )
        except ValueError:
            continue
    return rows or None


def short_kernel_name(mangled):
    """`void <unnamed>::prefill_kernel<(bool)1>(...)` -> `prefill_kernel<(bool)1>`."""
    name = mangled.strip().strip('"')
    name = name.split("(", 1)[0] if "(" in name.split("<", 1)[0] else name
    for prefix in ("void ", "<unnamed>::", "(anonymous namespace)::"):
        name = name.replace(prefix, "")
    depth, cut = 0, len(name)
    for index, char in enumerate(name):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "(" and depth == 0:
            cut = index
            break
    return name[:cut].strip()


def timeline_section(w):
    """Timeline observations, derived from whatever traces are on disk."""
    w("## Timeline observations (nsys)")
    w("")

    prefill_rows = nsys_kernel_stats("prefill")
    decode_rows = nsys_kernel_stats("decode")

    if prefill_rows is None and decode_rows is None:
        w("No nsys traces were found in this directory, so there is nothing to report")
        w("here. Regenerate them with:")
        w("")
        w("```bash")
        w("nsys profile -t cuda -o docs/profiles/prefill python bench/profile_workload.py prefill")
        w("nsys profile -t cuda -o docs/profiles/decode  python bench/profile_workload.py decode")
        w("```")
        w("")
        return

    w("Read back from `prefill.nsys-rep` and `decode.nsys-rep` in this directory,")
    w("collected with `nsys profile -t cuda` (no `--gpu-metrics-device`, which is")
    w("counter-gated). These are wall-clock kernel durations from the trace, not")
    w("counter measurements.")
    w("")
    w("| trace | kernel | launches | mean us | median us | min us | max us | spread (sd/mean) |")
    w("| :-- | :-- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, rows in (("prefill", prefill_rows), ("decode", decode_rows)):
        for row in rows or []:
            if row["instances"] < 5:  # skip one-off setup kernels
                continue
            spread = 100.0 * row["stddev_ns"] / row["avg_ns"] if row["avg_ns"] else 0.0
            w(
                f"| {label} | `{row['name']}` | {row['instances']} | "
                f"{row['avg_ns'] / 1e3:.2f} | {row['med_ns'] / 1e3:.2f} | "
                f"{row['min_ns'] / 1e3:.2f} | {row['max_ns'] / 1e3:.2f} | {spread:.1f}% |"
            )
    w("")

    if prefill_rows:
        main_kernel = prefill_rows[0]
        spread = (
            100.0 * main_kernel["stddev_ns"] / main_kernel["avg_ns"]
            if main_kernel["avg_ns"]
            else 0.0
        )
        w(
            f"- Prefill is one kernel per call: `{main_kernel['name']}` accounts for "
            f"{main_kernel['pct']:.1f}% of GPU time across {main_kernel['instances']} "
            f"launches, with {spread:.1f}% run-to-run spread."
        )

    if decode_rows:
        split = next((r for r in decode_rows if "split" in r["name"]), None)
        merge = next((r for r in decode_rows if "merge" in r["name"]), None)
        if split and merge:
            per_call = split["avg_ns"] + merge["avg_ns"]
            merge_pct = 100.0 * merge["avg_ns"] / per_call if per_call else 0.0
            w(
                f"- Decode is two kernels per call. The merge pass costs "
                f"{merge['avg_ns'] / 1e3:.2f} us against the split pass's "
                f"{split['avg_ns'] / 1e3:.2f} us, so it is {merge_pct:.1f}% of the "
                f"call at the chunk size the selection rule picked for this shape "
                f"(B=1, H=12, S=1024). That fraction is the price paid for "
                f"parallelism, and it grows as the chunk shrinks — which is why the "
                f"rule prefers the largest chunk that still fills the device rather "
                f"than the finest split available."
            )
    w("")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    torch.manual_seed(0)

    props = torch.cuda.get_device_properties(0)
    compute_peak = fp32_fma_peak_tflops(props)
    target_blocks = 2 * props.multi_processor_count

    out = []
    w = out.append

    w("# On-box performance analysis (T1.8)")
    w("")
    w("Generated by `bench/run_analysis.py` on the machine described below. Read the")
    w("two kinds of number here differently.")
    w("")
    w("**MEASURED** — observed on this machine, by a clock:")
    w("")
    w("- Kernel wall-clock times, from CUDA events (medians of repeated launches).")
    w("- Per-kernel durations and launch counts, read back from the nsys traces.")
    w("- Kernel resource footprints (registers/thread, shared memory/block) reported")
    w("  by `cudaFuncGetAttributes`.")
    w("")
    w("**MODELED** — arithmetic derived from those times plus the shapes, not observed:")
    w("")
    w("- TFLOP/s and GB/s, computed as analytic FLOP and byte counts divided by the")
    w("  measured time. They assume the kernel moves exactly the bytes the model says;")
    w("  real traffic including cache behaviour is not visible without counters.")
    w("- '% of peak' columns, which additionally assume a peak this card never")
    w("  advertises directly (see the compute-peak note below).")
    w("- Theoretical occupancy, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.")
    w("  It is an upper bound the hardware permits, **not** achieved occupancy.")
    w("")
    w("**NOT AVAILABLE HERE** — every counter-derived quantity. This machine denies")
    w("access to GPU performance counters (`ERR_NVGPUCTRPERM`, confirmed by smoke test;")
    w("it is a container, and the restriction is a per-host driver property that cannot")
    w("be lifted from inside). So SM and memory speed-of-light, **achieved** occupancy,")
    w("**measured** DRAM throughput and shared-memory bank-conflict counts are absent")
    w("rather than estimated. They are produced by `bench/profile_ncu.sh` on a")
    w("counter-capable RTX 3090 and land in [summary.md](summary.md), which is tracked")
    w("as open work (T1.9b) against a docs-only release. Until then, every claim in")
    w("this file about *which unit saturates* is a model, and is labelled as one.")
    w("")
    w("| | |")
    w("| :-- | :-- |")
    w(f"| GPU | {props.name} (sm_{props.major}{props.minor}, "
      f"{props.total_memory / 1024 ** 3:.0f} GB, {props.multi_processor_count} SMs) |")
    w(f"| Driver | {nvidia_smi('driver_version')} |")
    w(f"| CUDA runtime (torch) | {torch.version.cuda} |")
    w(f"| PyTorch | {torch.__version__} |")
    w(f"| Boost clock | {props.clock_rate / 1e6:.3f} GHz |")
    w(f"| fp32 FMA peak | {compute_peak:.1f} TFLOP/s |")
    w(f"| DRAM peak | {DRAM_PEAK_GBS:.0f} GB/s |")
    w("")
    w("**Which compute peak, and why.** The roofline below uses the fp32 FMA peak")
    w("(`SMs x 128 lanes x 2 flops x boost clock`), not the 71 TFLOP/s fp16 tensor-core")
    w("peak. Every multiply in these kernels is an `FFMA` on the CUDA cores: the contract")
    w("is fp32 accumulation, and the halves are converted once when a tile is staged, so")
    w("the tensor cores are never issued to. Comparing against a peak the kernel cannot")
    w("reach by construction would understate it; the tensor-core gap is a design")
    w("difference and is discussed against flash-attn in [../benchmarks.md](../benchmarks.md).")
    w("")

    w("## Kernel resources and theoretical occupancy")
    w("")
    w("From `flashattn_cuda.occupancy_report()`, which calls")
    w("`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on each kernel at the launch")
    w("configuration it actually uses.")
    w("")
    w("| kernel | threads/block | registers/thread | shared/block | max blocks/SM | theoretical occupancy |")
    w("| :-- | ---: | ---: | ---: | ---: | ---: |")
    for row in flashattn_cuda.occupancy_report():
        w(f"| `{row['kernel']}` | {row['threads_per_block']} | "
          f"{row['registers_per_thread']} | {row['shared_bytes_per_block']:,} B | "
          f"{row['max_active_blocks_per_sm']} | "
          f"{row['theoretical_occupancy'] * 100:.1f}% |")
    w("")
    w("Prefill is shared-memory limited, not register limited: 41,472 B per block against")
    w(f"the {props.shared_memory_per_multiprocessor // 1024} KB an SM can distribute allows")
    w("2 blocks, while the register file would allow more. Decode reaches full theoretical")
    w("occupancy per SM — its problem is having enough blocks to hand out, which is what")
    w("the split study below addresses.")
    w("")

    w("## Prefill roofline")
    w("")
    w("`FLOPs = 4 B H N^2 d` (halved for causal); `bytes = (3 reads + 1 write) B H N d 2`.")
    w("")
    w("| shape | ms | TFLOP/s | % of fp32 peak | GB/s | % of DRAM peak |")
    w("| :-- | ---: | ---: | ---: | ---: | ---: |")
    prefill_rows, _ = prefill_roofline(
        props,
        [
            (4, 12, 2048, True),
            (8, 12, 4096, True),
            (8, 12, 4096, False),
            (1, 12, 1024, True),
        ],
    )
    for r in prefill_rows:
        w(f"| {r['shape']} | {r['ms']:.3f} | {r['tflops']:.2f} | {r['pct_compute']:.1f}% | "
          f"{r['gbs']:.1f} | {r['pct_dram']:.1f}% |")
    w("")
    w("Prefill sits at a third of the fp32 compute peak and a few percent of the DRAM")
    w("peak, so it is compute-side bound — but not by the FMA pipe itself. The QK loop")
    w("reads 32 bytes of shared memory per 16 FMAs and the PV loop 48 bytes per 32, about")
    w("1.75 bytes per FMA, while an SM sustains 128 B/cycle of shared memory against 128")
    w("FMA/cycle. That ratio caps this blocking near 57% of fp32 peak before staging,")
    w("softmax bookkeeping and the three barriers per tile are paid for. Confirming that")
    w("shared memory is the limiter needs counters, and is a stated goal of T1.9.")
    w("")

    w("## Decode roofline and the split-size study")
    w("")
    w("`FLOPs = 4 B H S d` per step; `bytes = 2 B H S d 2` for the cache plus the")
    w("split workspace written by pass 1 and read by pass 2.")
    w("")
    w(f"Chunk selection targets `2 x SM count = {target_blocks}` blocks: the largest")
    w("power-of-two chunk in [128, 1024] that reaches it, falling back to 128 when no")
    w("chunk does.")
    w("")

    decode_shapes = [(1, 12, 512), (1, 12, 1024), (1, 12, 2048)]
    before = decode_roofline(props, decode_shapes, split_size=512)
    after = decode_roofline(props, decode_shapes)

    w("### Before / after: fixed Sk=512 versus adaptive")
    w("")
    w("| shape | Sk before | blocks | ms | GB/s | Sk after | blocks | ms | GB/s | speedup |")
    w("| :-- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b, a in zip(before, after):
        w(f"| {b['shape']} | {b['chunk']} | {b['blocks']} | {b['ms']:.4f} | {b['gbs']:.1f} | "
          f"{a['chunk']} | {a['blocks']} | {a['ms']:.4f} | {a['gbs']:.1f} | "
          f"{b['ms'] / a['ms']:.2f}x |")
    w("")

    w("### Adaptive decode across shapes")
    w("")
    w("| shape | chunk | blocks | ms | GB/s | % of DRAM peak | TFLOP/s | % of fp32 peak |")
    w("| :-- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in decode_roofline(
        props, [(1, 12, 512), (1, 12, 1024), (1, 12, 2048), (1, 12, 4096), (8, 12, 2048)]
    ):
        w(f"| {r['shape']} | {r['chunk']} | {r['blocks']} | {r['ms']:.4f} | {r['gbs']:.1f} | "
          f"{r['pct_dram']:.1f}% | {r['tflops']:.3f} | {r['pct_compute']:.2f}% |")
    w("")
    w("Decode does two FLOPs per byte of cache read, so it is a bandwidth problem with no")
    w("arithmetic to hide behind — the compute column is there to show how far from")
    w("compute-bound it is. What limits it at small batch is neither bandwidth nor the")
    w("access pattern (one warp reads one cached row as a single 128-byte coalesced")
    w("request) but the number of blocks in flight.")
    w("")
    w("### What the decode comparison baseline is, and is not")
    w("")
    w("The decode line in [../benchmarks.md](../benchmarks.md) is measured against a")
    w("**per-step eager SDPA loop**: one `scaled_dot_product_attention` call per decoded")
    w("token, over the whole cache, with a single-row query. That is the obvious way to")
    w("decode in PyTorch and it is a correct reference, but it is not a decode-optimised")
    w("kernel — it does not split the KV dimension, so it leaves unused exactly the")
    w("parallelism that the fixed-chunk version of this kernel used to leave unused. Beating")
    w("it is a low bar and should be read as such.")
    w("")
    w("The serious baselines are **flash-attn's dedicated decode path**")
    w("(`flash_attn_with_kvcache`) and **FlashDecoding**, both of which split the KV")
    w("dimension the way this kernel does and additionally run the arithmetic on tensor")
    w("cores. Benchmarking against those is deferred; until it happens, no claim is made")
    w("here about how this kernel compares to a state-of-the-art decode implementation.")
    w("")

    timeline_section(w)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out).rstrip() + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
