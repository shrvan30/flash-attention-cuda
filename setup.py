import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# The kernels are written for Ampere (RTX 3090, sm_86); override by exporting
# TORCH_CUDA_ARCH_LIST before building for a different card.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

this_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    "csrc/bindings.cpp",
    "csrc/prefill.cu",
    "csrc/decode.cu",
    "csrc/legacy/legacy_ops.cu",
    # v1 kernels, compiled unchanged so they stay benchmarkable from Python.
    "src/legacy/naive_attention.cu",
    "src/legacy/tiled_attention.cu",
    "src/legacy/fused_attention.cu",
]

setup(
    name="flashattn-cuda",
    version="2.0.0",
    description="Batched multi-head causal FlashAttention CUDA kernels (head_dim=64)",
    license="MIT",
    ext_modules=[
        CUDAExtension(
            name="flashattn_cuda",
            sources=[os.path.join(this_dir, s) for s in sources],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch"],
)
