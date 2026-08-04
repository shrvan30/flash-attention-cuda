"""Turn `ncu --csv --page raw` exports into docs/profiles/summary.md.

Called by bench/profile_ncu.sh after the capture. Kept separate so the parsing
can be tested without a GPU:

    python bench/ncu_extract.py --prefill p.csv --decode d.csv --out summary.md
"""

import argparse
import csv
import os

# Metric name -> (label, formatter). These are the T1.8/T1.9 required figures.
METRICS = [
    (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "SM speed of light",
        "{:.1f}%",
    ),
    (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory speed of light",
        "{:.1f}%",
    ),
    (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "Achieved occupancy",
        "{:.1f}%",
    ),
    ("dram__bytes.sum.per_second", "DRAM throughput", "{:.1f} GB/s"),
    (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "Shared load bank conflicts",
        "{:,.0f}",
    ),
    (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "Shared store bank conflicts",
        "{:,.0f}",
    ),
    ("gpu__time_duration.sum", "Kernel duration", "{:,.0f} ns"),
]

# ncu reports throughput in whatever unit fits; normalise to GB/s.
UNIT_SCALE = {
    "byte/second": 1e-9,
    "Kbyte/second": 1e-6,
    "Mbyte/second": 1e-3,
    "Gbyte/second": 1.0,
    "Tbyte/second": 1e3,
    "byte/s": 1e-9,
    "Gbyte/s": 1.0,
}


def parse(path):
    """{kernel name: {metric name: (value, unit)}} from an ncu raw CSV."""
    if not os.path.exists(path):
        raise SystemExit(f"missing ncu export: {path}")

    with open(path, newline="", encoding="utf-8") as handle:
        # ncu prints banner lines before the header; find the real header row.
        rows = list(csv.reader(handle))

    header_idx = None
    for i, row in enumerate(rows):
        if "Metric Name" in row and "Metric Value" in row:
            header_idx = i
            break
    if header_idx is None:
        raise SystemExit(f"{path}: no CSV header with a 'Metric Name' column")

    header = rows[header_idx]
    col = {name: idx for idx, name in enumerate(header)}
    kernel_col = col.get("Kernel Name", col.get("Function Name"))
    if kernel_col is None:
        raise SystemExit(f"{path}: no kernel-name column")
    name_col = col["Metric Name"]
    value_col = col["Metric Value"]
    unit_col = col.get("Metric Unit")

    out = {}
    for row in rows[header_idx + 1 :]:
        if len(row) <= max(kernel_col, name_col, value_col):
            continue
        kernel = row[kernel_col].strip()
        metric = row[name_col].strip()
        raw = row[value_col].strip().replace(",", "")
        if not kernel or not metric:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = row[unit_col].strip() if unit_col is not None and len(row) > unit_col else ""
        out.setdefault(kernel, {})[metric] = (value, unit)
    return out


def _scan_depth(name):
    """Yield (index, char, angle-bracket depth) over a demangled name."""
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
            yield i, ch, depth - 1
        elif ch == ">":
            depth = max(0, depth - 1)
            yield i, ch, depth
        else:
            yield i, ch, depth


def short_kernel_name(name):
    """`void (anonymous namespace)::foo<(bool)1>(args)` -> `foo<(bool)1>`.

    Template arguments are kept — they are what distinguishes the two prefill
    instantiations — so every bracket-aware step has to ignore punctuation that
    appears inside `<...>`.
    """
    name = name.replace("(anonymous namespace)::", "").strip()

    # Cut the argument list at the first parenthesis outside template brackets.
    for i, ch, depth in _scan_depth(name):
        if ch == "(" and depth == 0:
            name = name[:i]
            break
    name = name.strip()

    # Drop a leading return type: the last top-level space starts the real name.
    cut = 0
    for i, ch, depth in _scan_depth(name):
        if ch == " " and depth == 0:
            cut = i + 1
    name = name[cut:]

    # Drop namespace qualifiers that sit outside template arguments.
    last = 0
    for i, ch, depth in _scan_depth(name):
        if ch == ":" and depth == 0 and name[i : i + 2] == "::":
            last = i + 2
    return name[last:].strip() or "kernel"


def format_metric(metric, value, unit, fmt):
    if metric == "dram__bytes.sum.per_second":
        value *= UNIT_SCALE.get(unit, 1e-9)
    return fmt.format(value)


def kernel_table(metrics_by_kernel):
    kernels = list(metrics_by_kernel)
    lines = [
        "| metric | " + " | ".join(f"`{short_kernel_name(k)}`" for k in kernels) + " |",
        "| :-- | " + " | ".join("---:" for _ in kernels) + " |",
    ]
    for metric, label, fmt in METRICS:
        cells = []
        for kernel in kernels:
            entry = metrics_by_kernel[kernel].get(metric)
            cells.append(
                format_metric(metric, entry[0], entry[1], fmt) if entry else "—"
            )
        if all(cell == "—" for cell in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prefill = parse(args.prefill)
    decode = parse(args.decode)

    out = [
        "# Nsight Compute profiles (T1.9)",
        "",
        "Captured by `bench/profile_ncu.sh` with `ncu --set full` on a counter-capable",
        "RTX 3090. Reports: `ncu_prefill.ncu-rep`, `ncu_decode.ncu-rep`.",
        "",
        "The counter-free analysis taken on the development box — theoretical occupancy,",
        "the analytical roofline and the nsys timelines — is in [analysis.md](analysis.md);",
        "compare the achieved occupancy below against the theoretical figures there.",
        "",
        "## Prefill — N=2048, causal, B=4, H=12",
        "",
        kernel_table(prefill) if prefill else "_no prefill metrics found in the export_",
        "",
        "## Decode — S=1024, B=1, H=12",
        "",
        kernel_table(decode) if decode else "_no decode metrics found in the export_",
        "",
        "## Interpretation",
        "",
        "_To be written against the captured numbers: the top bottleneck for each kernel,",
        "and whether the shared-memory-bandwidth explanation in analysis.md survives the",
        "counter evidence (check SM SOL against the shared-load bank-conflict counts and",
        "the memory SOL split)._",
        "",
    ]

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out).rstrip() + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
