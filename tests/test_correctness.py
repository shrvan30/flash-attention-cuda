"""Correctness of the v2 kernels against PyTorch SDPA.

The reference is always computed in fp32 on upcast inputs and cast back to fp16,
so the comparison isolates the kernel's error rather than the reference's.
"""

import math

import pytest
import torch

import flashattn_cuda

HEAD_DIM = flashattn_cuda.head_dim
SCALE = 1.0 / math.sqrt(HEAD_DIM)
TOL = 2e-3

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


def rand(*shape, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(*shape, device="cuda", dtype=torch.float16)


def sdpa_reference(q, k, v, causal):
    """fp32 SDPA on upcast inputs, cast back to fp16."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, scale=SCALE
    )
    return out.half()


def decode_reference(q, k_cache, v_cache, seq_lens):
    """fp32 single-query attention over the valid part of each cache."""
    out = torch.empty_like(q)
    for b in range(q.shape[0]):
        n = int(seq_lens[b])
        scores = (q[b].float() @ k_cache[b, :, :n].float().transpose(-1, -2)) * SCALE
        out[b] = (torch.softmax(scores, dim=-1) @ v_cache[b, :, :n].float()).half()
    return out


def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("heads", [8, 12, 14])
@pytest.mark.parametrize("seq", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("causal", [True, False])
def test_prefill_matches_sdpa(batch, heads, seq, causal):
    q = rand(batch, heads, seq, HEAD_DIM, seed=0)
    k = rand(batch, heads, seq, HEAD_DIM)
    v = rand(batch, heads, seq, HEAD_DIM)

    out = flashattn_cuda.prefill(q, k, v, causal, SCALE)
    ref = sdpa_reference(q, k, v, causal)

    assert max_abs_diff(out, ref) <= TOL


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("seq", [1, 37, 65, 130, 1023])
@pytest.mark.parametrize("causal", [True, False])
def test_prefill_handles_ragged_sequence_lengths(seq, causal):
    """Sequence lengths that are not a multiple of the tile size."""
    q = rand(2, 5, seq, HEAD_DIM, seed=1)
    k = rand(2, 5, seq, HEAD_DIM)
    v = rand(2, 5, seq, HEAD_DIM)

    out = flashattn_cuda.prefill(q, k, v, causal, SCALE)
    ref = sdpa_reference(q, k, v, causal)

    assert max_abs_diff(out, ref) <= TOL


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("position", [1, 17, 511, 1024, 2047])
def test_decode_matches_causal_prefill_row(position):
    """Decoding position t must reproduce row t of the causal prefill output."""
    batch, heads, seq = 2, 12, 2048
    q = rand(batch, heads, seq, HEAD_DIM, seed=2)
    k = rand(batch, heads, seq, HEAD_DIM)
    v = rand(batch, heads, seq, HEAD_DIM)

    full = flashattn_cuda.prefill(q, k, v, True, SCALE)

    q_step = q[:, :, position : position + 1, :].contiguous()
    seq_lens = torch.full((batch,), position + 1, dtype=torch.int32, device="cuda")
    out = flashattn_cuda.decode(q_step, k, v, seq_lens, SCALE)

    assert max_abs_diff(out, full[:, :, position : position + 1, :]) <= TOL


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(
    "seq_lens",
    [
        [1, 2, 512, 4096],
        [513, 1023, 1024, 3000],
        [64, 65, 127, 129],
        [4096, 1, 2048, 7],
    ],
)
def test_decode_ragged_batch(seq_lens):
    """Every sequence in the batch has a different cache occupancy."""
    batch, heads, cache = len(seq_lens), 8, 4096
    q = rand(batch, heads, 1, HEAD_DIM, seed=3)
    k_cache = rand(batch, heads, cache, HEAD_DIM)
    v_cache = rand(batch, heads, cache, HEAD_DIM)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    out = flashattn_cuda.decode(q, k_cache, v_cache, lens, SCALE)
    ref = decode_reference(q, k_cache, v_cache, lens)

    assert max_abs_diff(out, ref) <= TOL


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("cache", [1, 63, 512, 513, 2048])
def test_decode_cache_lengths(cache):
    """Cache sizes around the split boundary (Sk = 512)."""
    batch, heads = 3, 12
    q = rand(batch, heads, 1, HEAD_DIM, seed=4)
    k_cache = rand(batch, heads, cache, HEAD_DIM)
    v_cache = rand(batch, heads, cache, HEAD_DIM)
    lens = torch.full((batch,), cache, dtype=torch.int32, device="cuda")

    out = flashattn_cuda.decode(q, k_cache, v_cache, lens, SCALE)
    ref = decode_reference(q, k_cache, v_cache, lens)

    assert max_abs_diff(out, ref) <= TOL


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("cache", [512, 1024, 2048])
def test_decode_is_invariant_to_split_size(cache):
    """The split-K chunk size is a scheduling choice, not a numerical one."""
    batch, heads = 2, 12
    q = rand(batch, heads, 1, HEAD_DIM, seed=7)
    k_cache = rand(batch, heads, cache, HEAD_DIM)
    v_cache = rand(batch, heads, cache, HEAD_DIM)
    lens = torch.tensor([cache, cache // 2 + 1], dtype=torch.int32, device="cuda")

    chosen = flashattn_cuda.decode(q, k_cache, v_cache, lens, SCALE)
    ref = decode_reference(q, k_cache, v_cache, lens)
    assert max_abs_diff(chosen, ref) <= TOL

    for split in (128, 256, 512, 1024):
        forced = flashattn_cuda._decode_with_split(
            q, k_cache, v_cache, lens, SCALE, split
        )
        assert max_abs_diff(forced, ref) <= TOL
        # Different chunkings reassociate the same sums, so require agreement at
        # the fp16 output level rather than bit-identity.
        assert max_abs_diff(forced, chosen) <= 1e-3


@pytest.mark.gpu
@requires_gpu
def test_decode_split_choice_targets_the_device():
    """The chosen chunk should fill the GPU when the shape allows it."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    target = 2 * sms

    for batch, heads, seq in [(1, 12, 2048), (8, 12, 2048), (4, 12, 1024)]:
        split = flashattn_cuda.choose_decode_split(seq, batch, heads, target)
        assert split in (128, 256, 512, 1024)
        blocks = batch * heads * -(-seq // split)
        assert blocks >= target
        # and it should be the largest such chunk: doubling it must fall short
        if split < 1024:
            bigger = batch * heads * -(-seq // (split * 2))
            assert bigger < target

    # Shapes too small to reach the target fall back to the finest split.
    assert flashattn_cuda.choose_decode_split(512, 1, 12, target) == 128


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
def test_prefill_rejects_bad_inputs():
    q = rand(1, 2, 8, HEAD_DIM, seed=5)

    with pytest.raises(RuntimeError, match="must be fp16"):
        flashattn_cuda.prefill(q.float(), q.float(), q.float(), True, SCALE)
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        flashattn_cuda.prefill(q.cpu(), q.cpu(), q.cpu(), True, SCALE)
    with pytest.raises(RuntimeError, match="must be 4-D"):
        flashattn_cuda.prefill(q[0], q[0], q[0], True, SCALE)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        flashattn_cuda.prefill(q[:, :, ::2], q, q, True, SCALE)
    with pytest.raises(RuntimeError, match="head_dim"):
        bad = rand(1, 2, 8, 32)
        flashattn_cuda.prefill(bad, bad, bad, True, SCALE)
    with pytest.raises(RuntimeError, match="identical shapes"):
        flashattn_cuda.prefill(q, q[:, :, :4].contiguous(), q, True, SCALE)


@pytest.mark.gpu
@requires_gpu
def test_decode_rejects_bad_inputs():
    q = rand(1, 2, 1, HEAD_DIM, seed=6)
    cache = rand(1, 2, 8, HEAD_DIM)
    lens = torch.tensor([4], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="exactly one query row"):
        flashattn_cuda.decode(cache, cache, cache, lens, SCALE)
    with pytest.raises(RuntimeError, match="seq_lens must be int32"):
        flashattn_cuda.decode(q, cache, cache, lens.long(), SCALE)
    with pytest.raises(RuntimeError, match="seq_lens must be 1-D"):
        flashattn_cuda.decode(q, cache, cache, lens.repeat(3), SCALE)
    with pytest.raises(RuntimeError, match="identical shapes"):
        flashattn_cuda.decode(q, cache, cache[:, :, :4].contiguous(), lens, SCALE)
