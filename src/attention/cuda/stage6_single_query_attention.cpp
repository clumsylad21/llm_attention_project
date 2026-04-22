#include <torch/extension.h>

torch::Tensor stage6_single_query_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache_t,
    torch::Tensor v_cache);

torch::Tensor stage6_single_query_attention_forward(
    torch::Tensor q,
    torch::Tensor k_cache_t,
    torch::Tensor v_cache) {
  if (!q.is_cuda() || !k_cache_t.is_cuda() || !v_cache.is_cuda()) {
    throw std::runtime_error("Stage 6 custom attention expects CUDA tensors.");
  }

  return stage6_single_query_attention_forward_cuda(q, k_cache_t, v_cache);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &stage6_single_query_attention_forward,
      "Stage 6 single-query tiled attention forward (CUDA)");
}
