#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr int TILE_TOKENS = 16;
constexpr int BLOCK_THREADS = 256;

template <typename scalar_t>
__global__ void single_query_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache_t,
    const scalar_t* __restrict__ v_cache,
    scalar_t* __restrict__ out,
    int batch,
    int heads,
    int visible_len,
    int head_dim) {
  int bh = blockIdx.x;
  int b = bh / heads;
  int h = bh % heads;
  int tid = threadIdx.x;

  const scalar_t* q_ptr = q + ((b * heads + h) * head_dim);
  const scalar_t* k_ptr = k_cache_t + ((b * heads + h) * head_dim * visible_len);
  const scalar_t* v_ptr = v_cache + ((b * heads + h) * visible_len * head_dim);
  scalar_t* out_ptr = out + ((b * heads + h) * head_dim);

  extern __shared__ float shared_mem[];
  float* k_tile = shared_mem;
  float* v_tile = k_tile + TILE_TOKENS * head_dim;
  float* reduction = v_tile + TILE_TOKENS * head_dim;
  float* weights = reduction + BLOCK_THREADS;
  float* scalars = weights + TILE_TOKENS;  // [0] = max_score, [1] = denominator

  float scale = rsqrtf(static_cast<float>(head_dim));

  if (tid == 0) {
    scalars[0] = -CUDART_INF_F;
  }
  __syncthreads();

  // ------------------------------------------------------------
  // Pass 1: find the stable softmax max(score)
  // ------------------------------------------------------------
  for (int tile_start = 0; tile_start < visible_len; tile_start += TILE_TOKENS) {
    int tile_len = min(TILE_TOKENS, visible_len - tile_start);

    for (int linear_idx = tid; linear_idx < tile_len * head_dim; linear_idx += BLOCK_THREADS) {
      int local_t = linear_idx / head_dim;
      int d = linear_idx % head_dim;
      k_tile[linear_idx] = static_cast<float>(k_ptr[d * visible_len + (tile_start + local_t)]);
    }
    __syncthreads();

    for (int local_t = 0; local_t < tile_len; ++local_t) {
      float partial = 0.0f;
      if (tid < head_dim) {
        partial = static_cast<float>(q_ptr[tid]) * k_tile[local_t * head_dim + tid];
      }

      reduction[tid] = partial;
      __syncthreads();

      for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
      }

      if (tid == 0) {
        float score = reduction[0] * scale;
        if (score > scalars[0]) {
          scalars[0] = score;
        }
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    scalars[1] = 0.0f;
  }
  __syncthreads();

  // ------------------------------------------------------------
  // Pass 2: accumulate exp(score - max) * V
  // Each thread owns one output feature dimension when tid < head_dim.
  // ------------------------------------------------------------
  float accum = 0.0f;

  for (int tile_start = 0; tile_start < visible_len; tile_start += TILE_TOKENS) {
    int tile_len = min(TILE_TOKENS, visible_len - tile_start);

    for (int linear_idx = tid; linear_idx < tile_len * head_dim; linear_idx += BLOCK_THREADS) {
      int local_t = linear_idx / head_dim;
      int d = linear_idx % head_dim;
      k_tile[linear_idx] = static_cast<float>(k_ptr[d * visible_len + (tile_start + local_t)]);
      v_tile[linear_idx] = static_cast<float>(v_ptr[(tile_start + local_t) * head_dim + d]);
    }
    __syncthreads();

    for (int local_t = 0; local_t < tile_len; ++local_t) {
      float partial = 0.0f;
      if (tid < head_dim) {
        partial = static_cast<float>(q_ptr[tid]) * k_tile[local_t * head_dim + tid];
      }

      reduction[tid] = partial;
      __syncthreads();

      for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
      }

      if (tid == 0) {
        float score = reduction[0] * scale;
        float weight = expf(score - scalars[0]);
        weights[local_t] = weight;
        scalars[1] += weight;
      }
      __syncthreads();

      if (tid < head_dim) {
        accum += weights[local_t] * v_tile[local_t * head_dim + tid];
      }
      __syncthreads();
    }
  }

  if (tid < head_dim) {
    out_ptr[tid] = static_cast<scalar_t>(accum / scalars[1]);
  }
}

}  // namespace

torch::Tensor stage6_single_query_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache_t,
    torch::Tensor v_cache) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k_cache_t.is_cuda(), "k_cache_t must be a CUDA tensor");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k_cache_t.is_contiguous(), "k_cache_t must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, 1, D]");
  TORCH_CHECK(k_cache_t.dim() == 4, "k_cache_t must have shape [B, H, D, T]");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape [B, H, T, D]");

  int64_t batch = q.size(0);
  int64_t heads = q.size(1);
  int64_t query_len = q.size(2);
  int64_t head_dim = q.size(3);

  TORCH_CHECK(query_len == 1, "Stage 6 kernel only supports single-query decode.");
  TORCH_CHECK(
      k_cache_t.size(0) == batch && k_cache_t.size(1) == heads,
      "k_cache_t batch/head dimensions must match q.");
  TORCH_CHECK(
      v_cache.size(0) == batch && v_cache.size(1) == heads,
      "v_cache batch/head dimensions must match q.");
  TORCH_CHECK(
      k_cache_t.size(2) == head_dim,
      "k_cache_t must have shape [B, H, D, T].");
  TORCH_CHECK(
      v_cache.size(3) == head_dim,
      "v_cache must have shape [B, H, T, D].");
  TORCH_CHECK(
      k_cache_t.size(3) == v_cache.size(2),
      "Visible sequence length must match between K and V.");
  TORCH_CHECK(
      head_dim > 0 && head_dim <= BLOCK_THREADS,
      "Stage 6 kernel currently supports 1 <= head_dim <= 256.");
  TORCH_CHECK(
      q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kFloat32,
      "Stage 6 kernel currently supports fp16 and fp32.");
  TORCH_CHECK(
      q.scalar_type() == k_cache_t.scalar_type() &&
          q.scalar_type() == v_cache.scalar_type(),
      "q, k_cache_t, and v_cache must have the same dtype.");

  auto q_contig = q.contiguous();
  auto k_contig = k_cache_t.contiguous();
  auto v_contig = v_cache.contiguous();

  auto out = torch::empty({batch, heads, 1, head_dim}, q.options());

  int visible_len = static_cast<int>(k_contig.size(3));
  int blocks = static_cast<int>(batch * heads);
  int threads = BLOCK_THREADS;
  size_t shared_bytes = static_cast<size_t>(
      (2 * TILE_TOKENS * head_dim + BLOCK_THREADS + TILE_TOKENS + 2) *
      sizeof(float));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      q_contig.scalar_type(),
      "stage6_single_query_attention_forward_cuda",
      ([&] {
        single_query_attention_kernel<scalar_t>
            <<<blocks, threads, shared_bytes, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                static_cast<int>(batch),
                static_cast<int>(heads),
                visible_len,
                static_cast<int>(head_dim));
      }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
