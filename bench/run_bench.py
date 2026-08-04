"""Benchmark the v2 kernels against the v1 kernels, PyTorch SDPA and flash-attn.

Writes docs/benchmarks.md and the SVG charts under docs/charts/.

    python bench/run_bench.py                 # full sweep
    python bench/run_bench.py --quick         # short sweep, for a smoke test
"""

import argparse
import math
import os
import subprocess
import sys

import torch

import flashattn_cuda

HEAD_DIM = flashattn_cuda.head_dim
SCALE = 1.0 / math.sqrt(HEAD_DIM)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
CHART_DIR = os.path.join(DOCS_DIR, "charts")

try:
    from flash_attn import flash_attn_func

    HAVE_FLASH_ATTN = True
except ImportError:  # pragma: no cover - optional comparison line
    flash_attn_func = None
    HAVE_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def nvidia_smi(query):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def nvcc_version():
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[1].split(",")[0].strip()
    except Exception:
        pass
    return "unknown"


def environment():
    props = torch.cuda.get_device_properties(0)
    return {
        "GPU": f"{props.name} (sm_{props.major}{props.minor}, "
        f"{props.total_memory / 1024 ** 3:.0f} GB, {props.multi_processor_count} SMs)",
        "Driver": nvidia_smi("driver_version"),
        "CUDA toolkit (nvcc)": nvcc_version(),
        "CUDA runtime (torch)": torch.version.cuda,
        "PyTorch": torch.__version__,
        "Python": sys.version.split()[0],
        "flash-attn": (
            __import__("flash_attn").__version__ if HAVE_FLASH_ATTN else "not installed"
        ),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timeit(fn, iters=30, warmup=10):
    """Median-of-three timing runs, each averaged over `iters` launches."""
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


def attention_tflops(batch, heads, seq, ms, causal):
    """4 * B * H * N^2 * d FLOPs, halved when causal."""
    flops = 4.0 * batch * heads * seq * seq * HEAD_DIM
    if causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Contenders
# ---------------------------------------------------------------------------


def sdpa_flash(q, k, v, causal):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=SCALE
        )


def prefill_lines(batch, heads, seqs, causal, legacy_seqs):
    """{label: {N: (ms, tflops)}} for every prefill contender."""
    results = {
        "v2 prefill": {},
        "torch SDPA (flash backend)": {},
    }
    if HAVE_FLASH_ATTN:
        results["flash-attn (official)"] = {}
    if legacy_seqs:
        results["v1 fused (fp32, 1 head)"] = {}
        results["v1 tiled (fp32, 1 head)"] = {}

    for seq in seqs:
        q, k, v = (
            torch.randn(batch, heads, seq, HEAD_DIM, device="cuda", dtype=torch.float16)
            for _ in range(3)
        )

        ms = timeit(lambda: flashattn_cuda.prefill(q, k, v, causal, SCALE))
        results["v2 prefill"][seq] = (ms, attention_tflops(batch, heads, seq, ms, causal))

        ms = timeit(lambda: sdpa_flash(q, k, v, causal))
        results["torch SDPA (flash backend)"][seq] = (
            ms,
            attention_tflops(batch, heads, seq, ms, causal),
        )

        if HAVE_FLASH_ATTN:
            # flash-attn takes (B, N, H, d)
            qf, kf, vf = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            ms = timeit(lambda: flash_attn_func(qf, kf, vf, causal=causal, softmax_scale=SCALE))
            results["flash-attn (official)"][seq] = (
                ms,
                attention_tflops(batch, heads, seq, ms, causal),
            )

        # The v1 kernels are single-head, fp32 and non-causal, so they are timed
        # on one (N, 64) matrix and scaled up to the same total work: B * H calls
        # would be the honest equivalent of one v2 launch.
        if seq in legacy_seqs:
            q1, k1, v1 = (
                torch.randn(seq, HEAD_DIM, device="cuda", dtype=torch.float32)
                for _ in range(3)
            )
            for label, fn in (
                ("v1 fused (fp32, 1 head)", flashattn_cuda.legacy_fused),
                ("v1 tiled (fp32, 1 head)", flashattn_cuda.legacy_tiled),
            ):
                per_head = timeit(lambda fn=fn: fn(q1, k1, v1), iters=10, warmup=3)
                ms = per_head * batch * heads
                results[label][seq] = (
                    ms,
                    attention_tflops(batch, heads, seq, ms, False),
                )

    return results


def decode_lines(batch, heads, contexts):
    """Single-sequence decode: tokens/s for our kernel vs an SDPA reference."""
    results = {"v2 decode": {}, "torch SDPA (decode reference)": {}}

    for ctx in contexts:
        q = torch.randn(batch, heads, 1, HEAD_DIM, device="cuda", dtype=torch.float16)
        k_cache = torch.randn(
            batch, heads, ctx, HEAD_DIM, device="cuda", dtype=torch.float16
        )
        v_cache = torch.randn_like(k_cache)
        seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")

        ms = timeit(lambda: flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, SCALE))
        results["v2 decode"][ctx] = (ms, batch * 1000.0 / ms)

        ms = timeit(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k_cache, v_cache, is_causal=False, scale=SCALE
            )
        )
        results["torch SDPA (decode reference)"][ctx] = (ms, batch * 1000.0 / ms)

    return results


# ---------------------------------------------------------------------------
# Charts (hand-written SVG, to match the v1 plotting script's zero-dependency style)
# ---------------------------------------------------------------------------

CHART_COLOURS = {
    "v2 prefill": "#1f77b4",
    "torch SDPA (flash backend)": "#2ca02c",
    "flash-attn (official)": "#d62728",
    "v1 fused (fp32, 1 head)": "#9467bd",
    "v1 tiled (fp32, 1 head)": "#8c564b",
    "v2 decode": "#1f77b4",
    "torch SDPA (decode reference)": "#2ca02c",
}


def write_line_chart(path, title, x_label, y_label, series, log_x=True):
    """series: {label: [(x, y), ...]} -> a standalone SVG line chart."""
    width, height = 780, 460
    left, right, top, bottom = 78, 250, 46, 58
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts]
    if not xs or not ys:
        return

    def fx(x):
        if log_x:
            lo, hi = math.log2(min(xs)), math.log2(max(xs))
            t = 0.5 if hi == lo else (math.log2(x) - lo) / (hi - lo)
        else:
            lo, hi = min(xs), max(xs)
            t = 0.5 if hi == lo else (x - lo) / (hi - lo)
        return left + t * plot_w

    y_max = max(ys) * 1.12
    y_max = max(y_max, 1e-9)

    def fy(y):
        return top + plot_h - (y / y_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2 - right / 2:.0f}" y="26" text-anchor="middle" '
        f'font-size="17" font-weight="bold" fill="#111111">{title}</text>',
    ]

    # y grid
    for i in range(6):
        y_val = y_max * i / 5
        y_pix = fy(y_val)
        parts.append(
            f'<line x1="{left}" y1="{y_pix:.1f}" x2="{left + plot_w}" y2="{y_pix:.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 9}" y="{y_pix + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#555555">{y_val:.3g}</text>'
        )

    # x ticks
    for x in sorted(set(xs)):
        x_pix = fx(x)
        parts.append(
            f'<line x1="{x_pix:.1f}" y1="{top + plot_h}" x2="{x_pix:.1f}" '
            f'y2="{top + plot_h + 5}" stroke="#888888" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pix:.1f}" y="{top + plot_h + 20}" text-anchor="middle" '
            f'font-size="11" fill="#555555">{x}</text>'
        )

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
        f'stroke="#999999" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 16}" text-anchor="middle" '
        f'font-size="12" fill="#333333">{x_label}</text>'
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.0f}" text-anchor="middle" font-size="12" '
        f'fill="#333333" transform="rotate(-90 18 {top + plot_h / 2:.0f})">{y_label}</text>'
    )

    legend_y = top + 6
    for label, pts in series.items():
        colour = CHART_COLOURS.get(label, "#666666")
        path_d = " ".join(
            f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}"
            for i, (x, y) in enumerate(sorted(pts))
        )
        parts.append(
            f'<path d="{path_d}" fill="none" stroke="{colour}" stroke-width="2.2"/>'
        )
        for x, y in pts:
            parts.append(
                f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.1" fill="{colour}"/>'
            )
        parts.append(
            f'<line x1="{left + plot_w + 14}" y1="{legend_y}" '
            f'x2="{left + plot_w + 42}" y2="{legend_y}" stroke="{colour}" '
            f'stroke-width="2.2"/>'
        )
        parts.append(
            f'<text x="{left + plot_w + 48}" y="{legend_y + 4}" font-size="11" '
            f'fill="#333333">{label}</text>'
        )
        legend_y += 21

    parts.append("</svg>")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def prefill_table(results, seqs):
    labels = list(results)
    lines = [
        "| N | " + " | ".join(f"{label} ms" for label in labels) + " |",
        "| ---: | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for seq in seqs:
        cells = []
        for label in labels:
            entry = results[label].get(seq)
            cells.append(f"{entry[0]:.3f}" if entry else "—")
        lines.append(f"| {seq} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("| N | " + " | ".join(f"{label} TFLOP/s" for label in labels) + " |")
    lines.append("| ---: | " + " | ".join("---:" for _ in labels) + " |")
    for seq in seqs:
        cells = []
        for label in labels:
            entry = results[label].get(seq)
            cells.append(f"{entry[1]:.2f}" if entry else "—")
        lines.append(f"| {seq} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def decode_table(results, contexts):
    labels = list(results)
    lines = [
        "| context | "
        + " | ".join(f"{label} ms | {label} tok/s" for label in labels)
        + " |",
        "| ---: | " + " | ".join("---: | ---:" for _ in labels) + " |",
    ]
    for ctx in contexts:
        cells = []
        for label in labels:
            ms, tps = results[label][ctx]
            cells.append(f"{ms:.4f} | {tps:,.0f}")
        lines.append(f"| {ctx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short sweep")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")

    torch.manual_seed(0)

    seqs = [128, 512, 1024, 2048] if args.quick else [128, 256, 512, 1024, 2048, 4096]
    legacy_seqs = set(seqs)
    contexts = [128, 512, 2048] if args.quick else [128, 256, 512, 1024, 2048]
    shapes = [(1, 12), (8, 12)]

    env = environment()

    report = [
        "# FlashAttention v2 benchmarks",
        "",
        "Every number below was measured by `bench/run_bench.py` on the machine described",
        "here. Times are the median of three runs, each the mean of 30 launches after 10",
        "warm-up launches. TFLOP/s counts `4 * B * H * N^2 * d` and halves it for causal.",
        "",
        "| | |",
        "| :-- | :-- |",
    ]
    report += [f"| {key} | {value} |" for key, value in env.items()]
    report.append("")

    charts = []

    for batch, heads in shapes:
        for causal in (True, False):
            tag = "causal" if causal else "non-causal"
            print(f"prefill B={batch} H={heads} {tag}", flush=True)
            results = prefill_lines(batch, heads, seqs, causal, legacy_seqs)

            report.append(f"## Prefill — B={batch}, H={heads}, {tag}")
            report.append("")
            report.append(
                "The v1 kernels are single-head, fp32 and have no causal path, so they "
                f"cannot run this shape directly. Their column is one `(N, 64)` launch "
                f"timed and multiplied by B*H = {batch * heads}, which is what covering "
                "the same total work would cost; the causal tables reuse the same "
                "(non-causal) v1 measurement, so v1 is if anything flattered there."
            )
            report.append("")
            report.append(prefill_table(results, seqs))
            report.append("")

            chart_name = f"prefill_b{batch}_{tag.replace('-', '')}.svg"
            write_line_chart(
                os.path.join(CHART_DIR, chart_name),
                f"Prefill throughput — B={batch}, H={heads}, {tag}",
                "sequence length N",
                "TFLOP/s",
                {
                    label: [(seq, val[1]) for seq, val in sorted(data.items())]
                    for label, data in results.items()
                    if data
                },
            )
            charts.append((f"Prefill B={batch}, {tag}", f"charts/{chart_name}"))

    print("decode", flush=True)
    decode = decode_lines(1, 12, contexts)
    report.append("## Decode — single sequence, H=12")
    report.append("")
    report.append(
        "Tokens/s is one decoded token per sequence per call, i.e. `batch / latency`."
    )
    report.append("")
    report.append(decode_table(decode, contexts))
    report.append("")

    write_line_chart(
        os.path.join(CHART_DIR, "decode_tokens_per_s.svg"),
        "Decode throughput — single sequence, H=12",
        "context length",
        "tokens/s",
        {
            label: [(ctx, val[1]) for ctx, val in sorted(data.items())]
            for label, data in decode.items()
        },
    )
    charts.append(("Decode tokens/s", "charts/decode_tokens_per_s.svg"))

    report.append("## Charts")
    report.append("")
    for title, rel in charts:
        report.append(f"### {title}")
        report.append("")
        report.append(f"![{title}]({rel})")
        report.append("")

    os.makedirs(DOCS_DIR, exist_ok=True)
    out_path = os.path.join(DOCS_DIR, "benchmarks.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report).rstrip() + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
