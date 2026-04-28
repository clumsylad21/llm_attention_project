#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

#ifndef STAGE6_TILE_TOKENS
#define STAGE6_TILE_TOKENS 32
#endif

constexpr int kMaxSupportedHeadDim = 256;
constexpr size_t kConservativeSharedLimitBytes = 48 * 1024;

// -----------------------------------------------------------------------------
// Kernel idea
// -----------------------------------------------------------------------------
// - one CUDA block handles one (batch, head)
// - one thread owns one output feature dimension when tid < head_dim
// - K/V are loaded in small time tiles into shared memory
// - we use an online softmax update so we do NOT need a separate max pass
//   and a second accumulation pass
//
// This is still intentionally small and educational, not production attention.
// -----------------------------------------------------------------------------

template <typename scalar_t, int BLOCK_THREADS>
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

  // Shared memory layout:
  // [ K tile | V tile | reduction scratch | softmax scalars ]
  float* k_tile = shared_mem;
  float* v_tile = k_tile + STAGE6_TILE_TOKENS * head_dim;
  float* reduction = v_tile + STAGE6_TILE_TOKENS * head_dim;
  float* softmax_state = reduction + BLOCK_THREADS;
  // softmax_state[0] = running_max
  // softmax_state[1] = running_sum
  // softmax_state[2] = rescale_old
  // softmax_state[3] = new_weight

  float accum = 0.0f;
  float scale = rsqrtf(static_cast<float>(head_dim));

  if (tid == 0) {
    softmax_state[0] = -CUDART_INF_F;
    softmax_state[1] = 0.0f;
    softmax_state[2] = 0.0f;
    softmax_state[3] = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < visible_len; tile_start += STAGE6_TILE_TOKENS) {
    int tile_len = min(STAGE6_TILE_TOKENS, visible_len - tile_start);

    // Load K and V tile into shared memory.
    for (int linear_idx = tid; linear_idx < tile_len * head_dim; linear_idx += BLOCK_THREADS) {
      int local_t = linear_idx / head_dim;
      int d = linear_idx % head_dim;

      k_tile[linear_idx] = static_cast<float>(k_ptr[d * visible_len + (tile_start + local_t)]);
      v_tile[linear_idx] = static_cast<float>(v_ptr[(tile_start + local_t) * head_dim + d]);
    }
    __syncthreads();

    for (int local_t = 0; local_t < tile_len; ++local_t) {
      // Dot product q · k_t using one reduction over feature dimensions.
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

        float old_max = softmax_state[0];
        float new_max = fmaxf(old_max, score);

        float rescale_old = (old_max == -CUDART_INF_F) ? 0.0f : expf(old_max - new_max);
        float new_weight = expf(score - new_max);

        softmax_state[0] = new_max;
        softmax_state[1] = softmax_state[1] * rescale_old + new_weight;
        softmax_state[2] = rescale_old;
        softmax_state[3] = new_weight;
      }
      __syncthreads();

      if (tid < head_dim) {
        accum = accum * softmax_state[2] + softmax_state[3] * v_tile[local_t * head_dim + tid];
      }
      __syncthreads();
    }
  }

  if (tid < head_dim) {
    out_ptr[tid] = static_cast<scalar_t>(accum / softmax_state[1]);
  }
}

int choose_block_threads(int head_dim) {
  if (head_dim <= 32) {
    return 32;
  }
  if (head_dim <= 64) {
    return 64;
  }
  if (head_dim <= 128) {
    return 128;
  }
  return 256;
}

template <typename scalar_t>
void launch_single_query_attention_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k_cache_t,
    const torch::Tensor& v_cache,
    torch::Tensor& out,
    int batch,
    int heads,
    int visible_len,
    int head_dim,
    cudaStream_t stream) {
  int blocks = batch * heads;
  int threads = choose_block_threads(head_dim);
  size_t shared_bytes =
      static_cast<size_t>(2 * STAGE6_TILE_TOKENS * head_dim + threads + 4) * sizeof(float);

  TORCH_CHECK(
      shared_bytes <= kConservativeSharedLimitBytes,
      "Stage 6 requested too much shared memory for this minimal kernel. "
      "Try a smaller tile size or smaller head_dim. "
      "shared_bytes=",
      shared_bytes,
      ", limit=",
      kConservativeSharedLimitBytes);

  const scalar_t* q_ptr = q.data_ptr<scalar_t>();
  const scalar_t* k_ptr = k_cache_t.data_ptr<scalar_t>();
  const scalar_t* v_ptr = v_cache.data_ptr<scalar_t>();
  scalar_t* out_ptr = out.data_ptr<scalar_t>();

  switch (threads) {
    case 32:
      single_query_attention_kernel<scalar_t, 32><<<blocks, 32, shared_bytes, stream>>>(
          q_ptr, k_ptr, v_ptr, out_ptr, batch, heads, visible_len, head_dim);
      break;
    case 64:
      single_query_attention_kernel<scalar_t, 64><<<blocks, 64, shared_bytes, stream>>>(
          q_ptr, k_ptr, v_ptr, out_ptr, batch, heads, visible_len, head_dim);
      break;
    case 128:
      single_query_attention_kernel<scalar_t, 128><<<blocks, 128, shared_bytes, stream>>>(
          q_ptr, k_ptr, v_ptr, out_ptr, batch, heads, visible_len, head_dim);
      break;
    default:
      single_query_attention_kernel<scalar_t, 256><<<blocks, 256, shared_bytes, stream>>>(
          q_ptr, k_ptr, v_ptr, out_ptr, batch, heads, visible_len, head_dim);
      break;
  }
}

void validate_stage6_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k_cache_t,
    const torch::Tensor& v_cache) {
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
  TORCH_CHECK(k_cache_t.size(2) == head_dim, "k_cache_t must have shape [B, H, D, T].");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache must have shape [B, H, T, D].");
  TORCH_CHECK(
      k_cache_t.size(3) == v_cache.size(2),
      "Visible sequence length must match between K and V.");
  TORCH_CHECK(
      head_dim > 0 && head_dim <= kMaxSupportedHeadDim,
      "Stage 6 kernel currently supports 1 <= head_dim <= 256.");
  TORCH_CHECK(k_cache_t.size(3) > 0, "Visible sequence length must be positive.");

  TORCH_CHECK(
      q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kFloat32,
      "Stage 6 kernel currently supports fp16 and fp32.");
  TORCH_CHECK(
      q.scalar_type() == k_cache_t.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
      "q, k_cache_t, and v_cache must have the same dtype.");
}

void launch_stage6_forward(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    torch::Tensor& out) {
  int batch = static_cast<int>(q_contig.size(0));
  int heads = static_cast<int>(q_contig.size(1));
  int visible_len = static_cast<int>(k_contig.size(3));
  int head_dim = static_cast<int>(q_contig.size(3));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      q_contig.scalar_type(),
      "stage6_single_query_attention_forward_cuda",
      ([&] {
        launch_single_query_attention_kernel<scalar_t>(
            q_contig,
            k_contig,
            v_contig,
            out,
            batch,
            heads,
            visible_len,
            head_dim,
            stream);
      }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor stage6_single_query_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache_t,
    torch::Tensor v_cache) {
  validate_stage6_inputs(q, k_cache_t, v_cache);

  auto q_contig = q.contiguous();
  auto k_contig = k_cache_t.contiguous();
  auto v_contig = v_cache.contiguous();

  auto out = torch::empty(
      {q_contig.size(0), q_contig.size(1), 1, q_contig.size(3)},
      q_contig.options());

  launch_stage6_forward(q_contig, k_contig, v_contig, out);
  return out;
}

void stage6_single_query_attention_forward_out_cuda(
    torch::Tensor q,
    torch::Tensor k_cache_t,
    torch::Tensor v_cache,
    torch::Tensor out) {
  validate_stage6_inputs(q, k_cache_t, v_cache);

  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.dim() == 4, "out must have shape [B, H, 1, D]");
  TORCH_CHECK(out.size(0) == q.size(0), "out batch dimension must match q");
  TORCH_CHECK(out.size(1) == q.size(1), "out head dimension must match q");
  TORCH_CHECK(out.size(2) == 1, "out query dimension must be 1");
  TORCH_CHECK(out.size(3) == q.size(3), "out feature dimension must match q");
  TORCH_CHECK(out.scalar_type() == q.scalar_type(), "out dtype must match q dtype");

  auto q_contig = q.contiguous();
  auto k_contig = k_cache_t.contiguous();
  auto v_contig = v_cache.contiguous();

  launch_stage6_forward(q_contig, k_contig, v_contig, out);
}
