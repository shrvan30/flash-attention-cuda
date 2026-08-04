#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

// The whole v2 stack is specialised for head_dim == 64. Every kernel below
// assumes it statically so that tile shapes, register blocking and the
// vectorised loads can be resolved at compile time.
constexpr int kHeadDim = 64;

// Prefill tile shape: Br query rows x Bc key rows per block, 128 threads.
constexpr int kPrefillBr = 64;
constexpr int kPrefillBc = 32;
constexpr int kPrefillThreads = 128;

// Decode split-K chunk: one block covers this many cached keys.
constexpr int kDecodeSplit = 512;
constexpr int kDecodeThreads = 128;

#define FA_CHECK_CUDA(x, name)                                                 \
  TORCH_CHECK((x).is_cuda(), name, " must be a CUDA tensor")

#define FA_CHECK_HALF(x, name)                                                 \
  TORCH_CHECK((x).scalar_type() == at::kHalf, name, " must be fp16 (got ",     \
              (x).scalar_type(), ")")

#define FA_CHECK_CONTIGUOUS(x, name)                                           \
  TORCH_CHECK((x).is_contiguous(), name, " must be contiguous row-major")

// (B, H, N, 64) fp16 CUDA tensor.
inline void fa_check_bhnd(const at::Tensor &t, const char *name) {
  FA_CHECK_CUDA(t, name);
  FA_CHECK_HALF(t, name);
  FA_CHECK_CONTIGUOUS(t, name);
  TORCH_CHECK(t.dim() == 4, name, " must be 4-D (B, H, N, ", kHeadDim,
              "), got ", t.dim(), "-D");
  TORCH_CHECK(t.size(3) == kHeadDim, name, " must have head_dim == ", kHeadDim,
              ", got ", t.size(3));
}

void launch_prefill(const at::Tensor &q, const at::Tensor &k,
                    const at::Tensor &v, at::Tensor &o, bool causal,
                    float scale);

void launch_decode(const at::Tensor &q, const at::Tensor &k_cache,
                   const at::Tensor &v_cache, const at::Tensor &seq_lens,
                   at::Tensor &o, float scale);
