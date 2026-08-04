"""Single-kernel workloads used by the profiling scripts.

    python bench/profile_workload.py prefill   # N=2048, causal, B=4, H=12
    python bench/profile_workload.py decode    # S=1024, B=1, H=12
"""

import math
import sys

import torch

import flashattn_cuda

HEAD_DIM = flashattn_cuda.head_dim
SCALE = 1.0 / math.sqrt(HEAD_DIM)
ITERS = 20


def prefill_workload():
    batch, heads, seq = 4, 12, 2048
    q, k, v = (
        torch.randn(batch, heads, seq, HEAD_DIM, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    for _ in range(5):
        flashattn_cuda.prefill(q, k, v, True, SCALE)
    torch.cuda.synchronize()
    for _ in range(ITERS):
        flashattn_cuda.prefill(q, k, v, True, SCALE)
    torch.cuda.synchronize()


def decode_workload():
    batch, heads, cache = 1, 12, 1024
    q = torch.randn(batch, heads, 1, HEAD_DIM, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(batch, heads, cache, HEAD_DIM, device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    seq_lens = torch.full((batch,), cache, dtype=torch.int32, device="cuda")
    for _ in range(5):
        flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, SCALE)
    torch.cuda.synchronize()
    for _ in range(ITERS):
        flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, SCALE)
    torch.cuda.synchronize()


WORKLOADS = {"prefill": prefill_workload, "decode": decode_workload}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in WORKLOADS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{'|'.join(WORKLOADS)}}}")
    torch.manual_seed(0)
    WORKLOADS[sys.argv[1]]()
