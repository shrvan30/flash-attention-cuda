#include <torch/extension.h>

#include "flash_common.h"

// ---------------------------------------------------------------------------
// prefill: batched multi-head attention over a full sequence.
//   q, k, v : (B, H, N, 64) fp16 CUDA, contiguous
//   returns o with the same shape/dtype.
// ---------------------------------------------------------------------------
at::Tensor prefill(const at::Tensor &q, const at::Tensor &k,
                   const at::Tensor &v, bool causal, double scale) {
  fa_check_bhnd(q, "q");
  fa_check_bhnd(k, "k");
  fa_check_bhnd(v, "v");

  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "prefill expects q, k and v to have identical shapes, got q=",
              q.sizes(), " k=", k.sizes(), " v=", v.sizes());
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "q, k and v must live on the same device");
  TORCH_CHECK(q.size(2) > 0, "sequence length must be > 0");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  at::Tensor o = at::empty_like(q);
  launch_prefill(q, k, v, o, causal, static_cast<float>(scale));
  return o;
}

// ---------------------------------------------------------------------------
// decode: one query row per sequence against a filled KV cache.
//   q             : (B, H, 1, 64) fp16 CUDA
//   k_cache/v_cache: (B, H, S_max, 64) fp16 CUDA
//   seq_lens      : (B,) int32 CUDA; sequence b attends rows [0, seq_lens[b])
//   returns o     : (B, H, 1, 64) fp16
// ---------------------------------------------------------------------------
at::Tensor decode(const at::Tensor &q, const at::Tensor &k_cache,
                  const at::Tensor &v_cache, const at::Tensor &seq_lens,
                  double scale) {
  fa_check_bhnd(q, "q");
  fa_check_bhnd(k_cache, "k_cache");
  fa_check_bhnd(v_cache, "v_cache");

  TORCH_CHECK(q.size(2) == 1, "decode expects exactly one query row, got ",
              q.size(2));
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache and v_cache must have identical shapes, got ",
              k_cache.sizes(), " and ", v_cache.sizes());
  TORCH_CHECK(q.size(0) == k_cache.size(0) && q.size(1) == k_cache.size(1),
              "q and the caches must agree on batch and head count, got q=",
              q.sizes(), " cache=", k_cache.sizes());
  TORCH_CHECK(k_cache.size(2) > 0, "cache length must be > 0");

  FA_CHECK_CUDA(seq_lens, "seq_lens");
  FA_CHECK_CONTIGUOUS(seq_lens, "seq_lens");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt,
              "seq_lens must be int32 (got ", seq_lens.scalar_type(), ")");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == q.size(0),
              "seq_lens must be 1-D of length B=", q.size(0), ", got ",
              seq_lens.sizes());
  TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device() &&
                  q.device() == seq_lens.device(),
              "q, caches and seq_lens must live on the same device");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  at::Tensor o = at::empty_like(q);
  launch_decode(q, k_cache, v_cache, seq_lens, o, static_cast<float>(scale));
  return o;
}

// ---------------------------------------------------------------------------
// v1 kernels, kept callable so the benchmark can plot them next to v2.
// They are single-head, fp32, non-causal: q, k, v are (N, 64) fp32.
// ---------------------------------------------------------------------------
at::Tensor legacy_naive(const at::Tensor &q, const at::Tensor &k,
                        const at::Tensor &v);
at::Tensor legacy_tiled(const at::Tensor &q, const at::Tensor &k,
                        const at::Tensor &v);
at::Tensor legacy_fused(const at::Tensor &q, const at::Tensor &k,
                        const at::Tensor &v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Batched multi-head causal FlashAttention kernels for head_dim=64";
  m.attr("head_dim") = kHeadDim;

  m.def("prefill", &prefill, "Full-sequence attention (fp16 in/out, fp32 accum)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"),
        py::arg("scale"));
  m.def("decode", &decode, "Single-query attention against a KV cache",
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("seq_lens"), py::arg("scale"));

  m.def("legacy_naive", &legacy_naive, "v1 naive kernel (N, 64) fp32");
  m.def("legacy_tiled", &legacy_tiled, "v1 tiled kernel (N, 64) fp32");
  m.def("legacy_fused", &legacy_fused, "v1 fused kernel (N, 64) fp32");
}
