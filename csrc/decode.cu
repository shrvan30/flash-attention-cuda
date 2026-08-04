// Single-query attention against a KV cache (the decode step).
//
// Prefill gets its parallelism from the query dimension; decode has exactly one
// query row per (batch, head), so the only axis left to spread across the GPU is
// the cached-key dimension. This is a split-K design:
//
//   pass 1  grid (splits, heads, batch): each block reduces one chunk of at most
//           kDecodeSplit cached keys into a partial (m, l, acc[64]) triple that
//           it writes to a workspace buffer.
//   pass 2  grid (heads, batch): merges the partials with the usual log-sum-exp
//           rule and writes the fp16 output.
//
// Inside a block the work is warp-per-key-row and lane-per-dim-pair: lane `t`
// holds head dims 2t and 2t+1, so a warp reading one cached row issues a single
// 128-byte coalesced request. Decode is bandwidth-bound, so the layout is chosen
// to keep every global access perfectly coalesced rather than to save FLOPs.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include "flash_common.h"

namespace {

constexpr int kThreads = kDecodeThreads;
constexpr int kWarps = kThreads / 32;
constexpr int kDimsPerLane = 2;  // one __half2 per lane covers head_dim=64

static_assert(kHeadDim == 64, "the v2 kernels are specialised for head_dim=64");
static_assert(kHeadDim == 32 * kDimsPerLane, "one warp must cover head_dim");
static_assert(kDecodeSplitMin % kWarps == 0, "warps split the chunk evenly");
static_assert(kDecodeSplitMin > 0 && kDecodeSplitMax >= kDecodeSplitMin,
              "split bounds must be a non-empty range");

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, off);
  }
  return x;
}

// ---------------------------------------------------------------------------
// Pass 1: reduce one chunk of cached keys into a partial (m, l, acc).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void decode_split_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k_cache,
    const __half *__restrict__ v_cache, const int *__restrict__ seq_lens,
    float *__restrict__ partial_acc, float *__restrict__ partial_ml,
    int max_seq, int n_splits, int split_size, float scale) {
  const int split = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int n_heads = gridDim.y;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int d0 = lane * kDimsPerLane;

  const size_t head_off =
      (static_cast<size_t>(batch) * n_heads + head) * static_cast<size_t>(max_seq) *
      kHeadDim;
  const size_t out_off =
      ((static_cast<size_t>(batch) * n_heads + head) * n_splits + split);

  const int seq_len = min(seq_lens[batch], max_seq);
  const int chunk_begin = split * split_size;
  const int chunk_end = min(seq_len, chunk_begin + split_size);

  __shared__ float s_acc[kWarps][kHeadDim];
  __shared__ float s_m[kWarps];
  __shared__ float s_l[kWarps];

  float m_i = -1e30f;
  float l_i = 0.f;
  float acc0 = 0.f;
  float acc1 = 0.f;

  if (chunk_begin < chunk_end) {
    // The single query row, split two dims per lane.
    const __half2 q_pair = *reinterpret_cast<const __half2 *>(
        q + (static_cast<size_t>(batch) * n_heads + head) * kHeadDim + d0);
    const float2 q_f = __half22float2(q_pair);

    for (int j = chunk_begin + warp; j < chunk_end; j += kWarps) {
      const size_t row = head_off + static_cast<size_t>(j) * kHeadDim + d0;
      const float2 k_f =
          __half22float2(*reinterpret_cast<const __half2 *>(k_cache + row));

      float s = warp_sum(q_f.x * k_f.x + q_f.y * k_f.y) * scale;

      const float m_new = fmaxf(m_i, s);
      const float rescale = __expf(m_i - m_new);
      const float p = __expf(s - m_new);
      m_i = m_new;
      l_i = l_i * rescale + p;

      const float2 v_f =
          __half22float2(*reinterpret_cast<const __half2 *>(v_cache + row));
      acc0 = fmaf(p, v_f.x, acc0 * rescale);
      acc1 = fmaf(p, v_f.y, acc1 * rescale);
    }
  }

  s_acc[warp][d0] = acc0;
  s_acc[warp][d0 + 1] = acc1;
  if (lane == 0) {
    s_m[warp] = m_i;
    s_l[warp] = l_i;
  }
  __syncthreads();

  // One warp folds the per-warp partials together and stores the block partial.
  if (threadIdx.x < kHeadDim) {
    const int d = threadIdx.x;
    float m_b = s_m[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      m_b = fmaxf(m_b, s_m[w]);
    }
    float l_b = 0.f;
    float acc_b = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float factor = __expf(s_m[w] - m_b);
      l_b = fmaf(s_l[w], factor, l_b);
      acc_b = fmaf(s_acc[w][d], factor, acc_b);
    }
    partial_acc[out_off * kHeadDim + d] = acc_b;
    if (d == 0) {
      partial_ml[out_off * 2] = m_b;
      partial_ml[out_off * 2 + 1] = l_b;
    }
  }
}

// ---------------------------------------------------------------------------
// Pass 2: log-sum-exp merge of the per-chunk partials.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kHeadDim) void decode_merge_kernel(
    const float *__restrict__ partial_acc, const float *__restrict__ partial_ml,
    __half *__restrict__ o, int n_splits) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int n_heads = gridDim.x;
  const int d = threadIdx.x;

  const size_t base =
      (static_cast<size_t>(batch) * n_heads + head) * n_splits;

  float m_all = -1e30f;
  for (int s = 0; s < n_splits; ++s) {
    if (partial_ml[(base + s) * 2 + 1] > 0.f) {
      m_all = fmaxf(m_all, partial_ml[(base + s) * 2]);
    }
  }

  float l_all = 0.f;
  float acc = 0.f;
  for (int s = 0; s < n_splits; ++s) {
    const float l_s = partial_ml[(base + s) * 2 + 1];
    if (l_s <= 0.f) {
      continue;  // chunk lies past this sequence's length
    }
    const float factor = __expf(partial_ml[(base + s) * 2] - m_all);
    l_all = fmaf(l_s, factor, l_all);
    acc = fmaf(partial_acc[(base + s) * kHeadDim + d], factor, acc);
  }

  const float inv_l = (l_all > 0.f) ? __frcp_rn(l_all) : 0.f;
  o[(static_cast<size_t>(batch) * n_heads + head) * kHeadDim + d] =
      __float2half(acc * inv_l);
}

}  // namespace

// A decode launch has only batch*heads*splits blocks, so the chunk size is what
// decides whether the GPU is filled. Pick the largest chunk that still reaches
// the block target: bigger chunks mean fewer partials for the merge pass, and
// the parallelism target is a floor, not something worth overshooting.
int choose_decode_split(int max_seq, int batch, int heads, int min_blocks) {
  const int64_t per_split = static_cast<int64_t>(batch) * heads;
  for (int split = kDecodeSplitMax; split > kDecodeSplitMin; split >>= 1) {
    const int64_t splits = (max_seq + split - 1) / split;
    if (per_split * splits >= min_blocks) {
      return split;
    }
  }
  return kDecodeSplitMin;
}

void launch_decode(const at::Tensor &q, const at::Tensor &k_cache,
                   const at::Tensor &v_cache, const at::Tensor &seq_lens,
                   at::Tensor &o, float scale, int split_size) {
  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int max_seq = static_cast<int>(k_cache.size(2));

  if (split_size <= 0) {
    // Two blocks per SM: enough to keep every SM fed and to overlap one block's
    // memory latency with another's, without splitting so finely that the merge
    // pass and the workspace traffic start to dominate.
    const int min_blocks =
        2 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    split_size = choose_decode_split(max_seq, batch, heads, min_blocks);
  }
  TORCH_CHECK(split_size % (kDecodeThreads / 32) == 0,
              "decode split size must be a multiple of the warp count, got ",
              split_size);

  const int n_splits = (max_seq + split_size - 1) / split_size;

  const auto workspace_opts = q.options().dtype(at::kFloat);
  at::Tensor partial_acc =
      at::empty({batch, heads, n_splits, kHeadDim}, workspace_opts);
  at::Tensor partial_ml = at::empty({batch, heads, n_splits, 2}, workspace_opts);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto *q_ptr = reinterpret_cast<const __half *>(q.const_data_ptr<at::Half>());
  const auto *k_ptr =
      reinterpret_cast<const __half *>(k_cache.const_data_ptr<at::Half>());
  const auto *v_ptr =
      reinterpret_cast<const __half *>(v_cache.const_data_ptr<at::Half>());
  auto *o_ptr = reinterpret_cast<__half *>(o.data_ptr<at::Half>());

  decode_split_kernel<<<dim3(n_splits, heads, batch), kThreads, 0, stream>>>(
      q_ptr, k_ptr, v_ptr, seq_lens.const_data_ptr<int>(),
      partial_acc.data_ptr<float>(), partial_ml.data_ptr<float>(), max_seq,
      n_splits, split_size, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  decode_merge_kernel<<<dim3(heads, batch), kHeadDim, 0, stream>>>(
      partial_acc.const_data_ptr<float>(), partial_ml.const_data_ptr<float>(),
      o_ptr, n_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode_occupancy(std::vector<OccupancyEntry> &out) {
  const auto *props = at::cuda::getCurrentDeviceProperties();

  auto record = [&](const char *name, const void *func, int threads) {
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, func));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, func,
                                                                 threads, 0));
    out.push_back({name, threads, attr.numRegs,
                   static_cast<int>(attr.sharedSizeBytes), blocks,
                   static_cast<double>(blocks) * threads /
                       props->maxThreadsPerMultiProcessor});
  };

  record("decode_split_kernel", reinterpret_cast<const void *>(decode_split_kernel),
         kDecodeThreads);
  record("decode_merge_kernel", reinterpret_cast<const void *>(decode_merge_kernel),
         kHeadDim);
}
