#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

// The whole v2 stack is specialised for head_dim == 64. Every kernel below
// assumes it statically so that tile shapes, register blocking and the
// vectorised loads can be resolved at compile time.
constexpr int kHeadDim = 64;

// Prefill tile shape: Br query rows x Bc key rows per block, 128 threads.
constexpr int kPrefillBr = 64;
constexpr int kPrefillBc = 32;
constexpr int kPrefillThreads = 128;

// Decode split-K chunk: one block covers this many cached keys. The size is
// chosen per launch (see choose_decode_split) between these bounds, because a
// fixed chunk either starves the GPU at small batch or over-splits at large.
constexpr int kDecodeSplitMin = 128;
constexpr int kDecodeSplitMax = 1024;
constexpr int kDecodeThreads = 128;

// Largest power-of-two chunk in [kDecodeSplitMin, kDecodeSplitMax] for which
// batch * heads * ceil(max_seq / chunk) reaches `min_blocks`; falls back to the
// smallest chunk (most parallelism) when no size reaches it.
int choose_decode_split(int max_seq, int batch, int heads, int min_blocks);

// One row of flashattn_cuda.occupancy_report().
struct OccupancyEntry {
  const char *name;
  int threads_per_block;
  int registers_per_thread;
  int shared_bytes_per_block;
  int max_active_blocks_per_sm;
  double theoretical_occupancy;
};

void prefill_occupancy(std::vector<OccupancyEntry> &out);
void decode_occupancy(std::vector<OccupancyEntry> &out);

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

// split_size == 0 selects the chunk automatically; a positive value forces it
// (used by the benchmark to compare split strategies).
void launch_decode(const at::Tensor &q, const at::Tensor &k_cache,
                   const at::Tensor &v_cache, const at::Tensor &seq_lens,
                   at::Tensor &o, float scale, int split_size);
