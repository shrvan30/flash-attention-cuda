// Batched multi-head FlashAttention prefill for head_dim == 64.
//
// One block owns a Br x head_dim tile of the output for a single (batch, head)
// pair and streams the whole K/V sequence past it in Bc-row tiles, keeping the
// running softmax statistics (m, l) and the output accumulator in registers.
//
// Thread layout inside a block (128 threads = 8 column groups x 16 row groups):
//
//   scores  S : each thread owns 4 query rows x 4 key columns
//   output  O : each thread owns the same 4 query rows x 8 head dims
//
// Both matmuls are register-blocked so that every value pulled out of shared
// memory feeds several FMAs. Q/K live in shared memory transposed (dim-major)
// so a thread can grab the four rows it owns for one head dim with a single
// 128-bit load; P has to make a round trip through shared memory because the
// PV product needs a different slice of it than the QK product produced.
//
// Shared tiles are kept in fp32: the halves are converted once when the tile is
// staged, instead of once per use, which keeps the fp32-accumulate contract
// without paying for a conversion inside the inner loop.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include "flash_common.h"

namespace {

constexpr int kBr = kPrefillBr;          // query rows per block
constexpr int kBc = kPrefillBc;          // key rows per tile
constexpr int kThreads = kPrefillThreads;

constexpr int kRowsPerThread = 4;        // query rows owned by one thread
constexpr int kColsPerThread = 4;        // key columns owned by one thread
constexpr int kDimsPerThread = 8;        // head dims owned by one thread
constexpr int kColGroups = kBc / kColsPerThread;   // 8
constexpr int kRowGroups = kBr / kRowsPerThread;   // 16
constexpr int kPsStride = kBr + 4;       // padded to break bank conflicts

static_assert(kHeadDim == 64, "the v2 kernels are specialised for head_dim=64");
static_assert(kRowGroups * kColGroups == kThreads, "thread grid must cover the tile");
static_assert(kColGroups * kDimsPerThread == kHeadDim, "column groups must cover head_dim");
static_assert(kHeadDim % 8 == 0, "vectorised loads move 8 halves at a time");

// Column swizzle for the dim-major tiles. Staging a tile writes one row of the
// source (fixed row index, eight consecutive head dims) which would otherwise
// land in a single bank; XOR-ing the column with the head-dim chunk spreads
// those writes across banks while leaving 4-element groups (and therefore
// 128-bit loads) intact.
__device__ __forceinline__ int swizzle(int col, int dd) {
  return col ^ (4 * (dd >> 3));
}

// Stage `tile_rows` rows of a row-major (·, 64) fp16 matrix into a dim-major
// fp32 shared tile of shape [64][tile_rows]. Rows past `rows_valid` are zeroed.
template <int TILE_ROWS>
__device__ __forceinline__ void stage_dim_major(const __half *__restrict__ src,
                                                int row0, int rows_valid,
                                                float *__restrict__ smem,
                                                int tid) {
  constexpr int kPairs = TILE_ROWS * (kHeadDim / 8);
#pragma unroll
  for (int idx = tid; idx < kPairs; idx += kThreads) {
    const int r = idx >> 3;
    const int chunk = idx & 7;
    const int dd0 = chunk * 8;

    float4 raw = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < rows_valid) {
      raw = *reinterpret_cast<const float4 *>(
          src + static_cast<size_t>(row0 + r) * kHeadDim + dd0);
    }
    const __half2 *packed = reinterpret_cast<const __half2 *>(&raw);
    const int col = r ^ (4 * chunk);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 f = __half22float2(packed[t]);
      smem[(dd0 + 2 * t) * TILE_ROWS + col] = f.x;
      smem[(dd0 + 2 * t + 1) * TILE_ROWS + col] = f.y;
    }
  }
}

// Stage `tile_rows` rows into a row-major fp32 shared tile of shape [rows][64].
template <int TILE_ROWS>
__device__ __forceinline__ void stage_row_major(const __half *__restrict__ src,
                                                int row0, int rows_valid,
                                                float *__restrict__ smem,
                                                int tid) {
  constexpr int kPairs = TILE_ROWS * (kHeadDim / 8);
#pragma unroll
  for (int idx = tid; idx < kPairs; idx += kThreads) {
    const int r = idx >> 3;
    const int chunk = idx & 7;
    const int dd0 = chunk * 8;

    float4 raw = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < rows_valid) {
      raw = *reinterpret_cast<const float4 *>(
          src + static_cast<size_t>(row0 + r) * kHeadDim + dd0);
    }
    const __half2 *packed = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 f = __half22float2(packed[t]);
      smem[r * kHeadDim + dd0 + 2 * t] = f.x;
      smem[r * kHeadDim + dd0 + 2 * t + 1] = f.y;
    }
  }
}

// Reduce a per-row value across the 8 column groups that share a row group.
// Those threads occupy 8 consecutive lanes, so the butterfly stays inside a warp.
template <bool IS_MAX>
__device__ __forceinline__ float row_reduce(float x) {
#pragma unroll
  for (int off = 1; off < kColGroups; off <<= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, x, off);
    x = IS_MAX ? fmaxf(x, other) : x + other;
  }
  return x;
}

template <bool CAUSAL>
__global__ __launch_bounds__(kThreads) void prefill_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, __half *__restrict__ o, int N, float scale) {
  const int q_tile = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int n_heads = gridDim.y;

  const size_t head_off =
      (static_cast<size_t>(batch) * n_heads + head) * static_cast<size_t>(N) *
      kHeadDim;
  q += head_off;
  k += head_off;
  v += head_off;
  o += head_off;

  const int tid = threadIdx.x;
  const int row_group = tid / kColGroups;
  const int col_group = tid % kColGroups;
  const int i0 = row_group * kRowsPerThread;  // first query row of this thread
  const int j0 = col_group * kColsPerThread;  // first key column of this thread
  const int d0 = col_group * kDimsPerThread;  // first head dim of this thread

  __shared__ float q_tile_s[kHeadDim][kBr];   // [dd][i], swizzled columns
  __shared__ float k_tile_s[kHeadDim][kBc];   // [dd][j], swizzled columns
  __shared__ float v_tile_s[kBc][kHeadDim];   // [j][dd]
  __shared__ float p_tile_s[kBc][kPsStride];  // [j][i]

  const int q_row0 = q_tile * kBr;
  stage_dim_major<kBr>(q, q_row0, min(kBr, N - q_row0), &q_tile_s[0][0], tid);

  float acc[kRowsPerThread][kDimsPerThread];
  float m_i[kRowsPerThread];
  float l_i[kRowsPerThread];
#pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r) {
    m_i[r] = -1e30f;  // finite so the first rescale factor is 0, not NaN
    l_i[r] = 0.f;
#pragma unroll
    for (int d = 0; d < kDimsPerThread; ++d) {
      acc[r][d] = 0.f;
    }
  }

  const int n_kv_tiles = (N + kBc - 1) / kBc;

  for (int t = 0; t < n_kv_tiles; ++t) {
    const int k_row0 = t * kBc;
    const int rows_valid = min(kBc, N - k_row0);

    __syncthreads();  // previous tile's PV has finished reading V/P
    stage_dim_major<kBc>(k, k_row0, rows_valid, &k_tile_s[0][0], tid);
    stage_row_major<kBc>(v, k_row0, rows_valid, &v_tile_s[0][0], tid);
    __syncthreads();

    // ---- S = Q K^T -----------------------------------------------------
    float s[kRowsPerThread][kColsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
#pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) {
        s[r][c] = 0.f;
      }
    }
#pragma unroll 8
    for (int dd = 0; dd < kHeadDim; ++dd) {
      const float4 qv = *reinterpret_cast<const float4 *>(
          &q_tile_s[dd][swizzle(i0, dd)]);
      const float4 kv = *reinterpret_cast<const float4 *>(
          &k_tile_s[dd][swizzle(j0, dd)]);
      const float qr[kRowsPerThread] = {qv.x, qv.y, qv.z, qv.w};
      const float kc[kColsPerThread] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
#pragma unroll
        for (int c = 0; c < kColsPerThread; ++c) {
          s[r][c] = fmaf(qr[r], kc[c], s[r][c]);
        }
      }
    }

    // ---- scale, mask, online softmax ------------------------------------
    float p[kRowsPerThread][kColsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      const int q_idx = q_row0 + i0 + r;
      float row_max = -1e30f;
#pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) {
        const int k_idx = k_row0 + j0 + c;
        const bool keep = (k_idx < N) && (!CAUSAL || k_idx <= q_idx);
        s[r][c] = keep ? s[r][c] * scale : -INFINITY;
        row_max = fmaxf(row_max, s[r][c]);
      }
      row_max = row_reduce<true>(row_max);

      const float m_new = fmaxf(m_i[r], row_max);
      const float rescale = __expf(m_i[r] - m_new);
      m_i[r] = m_new;

      float row_sum = 0.f;
#pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) {
        p[r][c] = __expf(s[r][c] - m_new);
        row_sum += p[r][c];
      }
      row_sum = row_reduce<false>(row_sum);

      l_i[r] = l_i[r] * rescale + row_sum;
#pragma unroll
      for (int d = 0; d < kDimsPerThread; ++d) {
        acc[r][d] *= rescale;
      }
    }

    // ---- O += P V --------------------------------------------------------
    // P is regrouped through shared memory: the QK blocking hands each thread a
    // 4x4 patch of P, while PV needs whole columns of it.
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
#pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) {
        p_tile_s[j0 + c][i0 + r] = p[r][c];
      }
    }
    __syncthreads();

#pragma unroll 4
    for (int j = 0; j < kBc; ++j) {
      const float4 pv =
          *reinterpret_cast<const float4 *>(&p_tile_s[j][i0]);
      const float4 v0 = *reinterpret_cast<const float4 *>(&v_tile_s[j][d0]);
      const float4 v1 = *reinterpret_cast<const float4 *>(&v_tile_s[j][d0 + 4]);
      const float pr[kRowsPerThread] = {pv.x, pv.y, pv.z, pv.w};
      const float vd[kDimsPerThread] = {v0.x, v0.y, v0.z, v0.w,
                                        v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
#pragma unroll
        for (int d = 0; d < kDimsPerThread; ++d) {
          acc[r][d] = fmaf(pr[r], vd[d], acc[r][d]);
        }
      }
    }
  }

  // ---- epilogue: normalise and write 16 bytes per row --------------------
#pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r) {
    const int q_idx = q_row0 + i0 + r;
    if (q_idx >= N) {
      continue;
    }
    const float inv_l = (l_i[r] > 0.f) ? __frcp_rn(l_i[r]) : 0.f;
    __half out[kDimsPerThread];
#pragma unroll
    for (int d = 0; d < kDimsPerThread; ++d) {
      out[d] = __float2half(acc[r][d] * inv_l);
    }
    *reinterpret_cast<float4 *>(o + static_cast<size_t>(q_idx) * kHeadDim + d0) =
        *reinterpret_cast<const float4 *>(out);
  }
}

}  // namespace

void launch_prefill(const at::Tensor &q, const at::Tensor &k,
                    const at::Tensor &v, at::Tensor &o, bool causal,
                    float scale) {
  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int seq = static_cast<int>(q.size(2));

  const dim3 grid((seq + kBr - 1) / kBr, heads, batch);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto *q_ptr = reinterpret_cast<const __half *>(q.const_data_ptr<at::Half>());
  const auto *k_ptr = reinterpret_cast<const __half *>(k.const_data_ptr<at::Half>());
  const auto *v_ptr = reinterpret_cast<const __half *>(v.const_data_ptr<at::Half>());
  auto *o_ptr = reinterpret_cast<__half *>(o.data_ptr<at::Half>());

  if (causal) {
    prefill_kernel<true><<<grid, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr,
                                                        o_ptr, seq, scale);
  } else {
    prefill_kernel<false><<<grid, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr,
                                                         o_ptr, seq, scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
