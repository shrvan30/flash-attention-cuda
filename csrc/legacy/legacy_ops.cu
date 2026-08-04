// Thin wrappers that make the v1 kernels callable from Python so the v2
// benchmark can plot them on the same axes. The kernels themselves are the
// unchanged v1 sources under src/legacy/.

#include <ATen/ATen.h>
#include <torch/extension.h>

#include "../flash_common.h"

void launch_naive_attention(const float *d_Q, const float *d_K,
                            const float *d_V, float *d_O, int N, int d);
void launch_tiled_attention(const float *d_Q, const float *d_K,
                            const float *d_V, float *d_O, int N, int d);
void launch_fused_attention(const float *d_Q, const float *d_K,
                            const float *d_V, float *d_O, int N, int d);

namespace {

void check_legacy_inputs(const at::Tensor &q, const at::Tensor &k,
                         const at::Tensor &v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "legacy kernels expect CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat,
              "legacy kernels are fp32 only");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "legacy kernels expect contiguous tensors");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2,
              "legacy kernels are single-head: expected (N, ", kHeadDim,
              ") tensors");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "legacy kernels expect identical q/k/v shapes");
  TORCH_CHECK(q.size(1) == kHeadDim, "legacy kernels expect head_dim == ",
              kHeadDim);
}

} // namespace

#define FA_LEGACY_OP(name, launcher)                                           \
  at::Tensor name(const at::Tensor &q, const at::Tensor &k,                    \
                  const at::Tensor &v) {                                       \
    check_legacy_inputs(q, k, v);                                              \
    at::Tensor o = at::empty_like(q);                                          \
    launcher(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),    \
             o.data_ptr<float>(), static_cast<int>(q.size(0)), kHeadDim);      \
    return o;                                                                  \
  }

FA_LEGACY_OP(legacy_naive, launch_naive_attention)
FA_LEGACY_OP(legacy_tiled, launch_tiled_attention)
FA_LEGACY_OP(legacy_fused, launch_fused_attention)
