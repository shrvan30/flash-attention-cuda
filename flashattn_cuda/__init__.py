"""Batched multi-head causal FlashAttention kernels for head_dim = 64."""

# The compiled extension links against libc10/libtorch, which are only on the
# loader's path once torch itself has been imported. Importing it here means
# `import flashattn_cuda` works on its own, in any order.
import torch as _torch  # noqa: F401

from ._C import (  # noqa: F401
    decode,
    head_dim,
    legacy_fused,
    legacy_naive,
    legacy_tiled,
    prefill,
)

__all__ = [
    "prefill",
    "decode",
    "head_dim",
    "legacy_naive",
    "legacy_tiled",
    "legacy_fused",
]

__version__ = "2.0.0"
