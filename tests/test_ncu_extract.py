"""Tests for the ncu report parser.

The capture itself runs on a different machine, so these guard the parsing on
this one — a formatting bug found after the profiling rental has ended is a bug
found too late. No GPU required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench"))

import ncu_extract  # noqa: E402


@pytest.mark.parametrize(
    "mangled,expected",
    [
        (
            "void (anonymous namespace)::prefill_kernel<(bool)1>(const __half *, int)",
            "prefill_kernel<(bool)1>",
        ),
        (
            "void (anonymous namespace)::prefill_kernel<(bool)0>(const __half *, int)",
            "prefill_kernel<(bool)0>",
        ),
        (
            "(anonymous namespace)::decode_split_kernel(const __half *)",
            "decode_split_kernel",
        ),
        (
            "(anonymous namespace)::decode_merge_kernel(const float *, int)",
            "decode_merge_kernel",
        ),
        ("void at::native::foo<int, (bool)1>(x)", "foo<int, (bool)1>"),
        ("simple_kernel", "simple_kernel"),
    ],
)
def test_short_kernel_name(mangled, expected):
    assert ncu_extract.short_kernel_name(mangled) == expected


def test_prefill_instantiations_stay_distinguishable():
    causal = ncu_extract.short_kernel_name(
        "void (anonymous namespace)::prefill_kernel<(bool)1>(const __half *)"
    )
    full = ncu_extract.short_kernel_name(
        "void (anonymous namespace)::prefill_kernel<(bool)0>(const __half *)"
    )
    assert causal != full


CSV = '''==PROF== Connected to process 1
"ID","Process ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","(anonymous namespace)::decode_split_kernel(const __half *)","Speed Of Light","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","12.3"
"0","1","(anonymous namespace)::decode_split_kernel(const __half *)","Occupancy","sm__warps_active.avg.pct_of_peak_sustained_active","%","74.2"
"0","1","(anonymous namespace)::decode_split_kernel(const __half *)","Memory","dram__bytes.sum.per_second","Tbyte/second","0.4103"
"0","1","(anonymous namespace)::decode_split_kernel(const __half *)","Memory","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum","","98304"
'''


def test_parse_and_render(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(CSV, encoding="utf-8")

    parsed = ncu_extract.parse(str(csv_path))
    assert len(parsed) == 1
    metrics = next(iter(parsed.values()))
    assert metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"][0] == 12.3

    table = ncu_extract.kernel_table(parsed)
    assert "`decode_split_kernel`" in table
    assert "12.3%" in table
    assert "74.2%" in table
    # Tbyte/second must be normalised to GB/s, not printed as 0.4
    assert "410.3 GB/s" in table
    # thousands separator on raw counts
    assert "98,304" in table
    # metrics absent from the export are dropped, not rendered as zero
    assert "Memory speed of light" not in table


def test_parse_rejects_a_csv_without_metrics(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text('"a","b"\n"1","2"\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        ncu_extract.parse(str(bad))
