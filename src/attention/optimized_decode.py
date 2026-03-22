import math
from dataclasses import dataclass

import torch

from src.attention.decode import ProjectionWeights
from src.attention.optimized_kv_cache import OptimizedKVCache


@dataclass
class FusedProjectionWeights:
    """
    Fused projection weights for one-matmul QKV projection.

    W_qkv shape:
        [model_dim, 3 * model_dim]
    """
    W_qkv: torch.Tensor
    model_dim: int
    num_heads: int
    head_dim: int


def build_fused_projection_weights_from_separate(
    weights: ProjectionWeights,
    heads: int,
    head_dim: int,
) -> FusedProjectionWeights:
    """
    Build one fused [Wq | Wk | Wv] matrix from the existing separate weights.
    """
    Wq = weights.w_q
    Wk = weights.w_k
    Wv = weights.w_v

    model_dim = Wq.shape[0]

    if model_dim != heads * head_dim:
        raise ValueError(
            f"model_dim={model_dim} does not match heads*head_dim={heads * head_dim}"
        )

    W_qkv = torch.cat([Wq, Wk, Wv], dim=-1).contiguous()

    return FusedProjectionWeights(
        W_qkv=W_qkv,
        model_dim=model_dim,
        num_heads=heads,
        head_dim=head_dim,
    )


def reshape_to_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Convert [B, S, model_dim] -> [B, H, S, D]
    """
    batch_size, seq_len, model_dim = x.shape
    expected_model_dim = num_heads * head_dim

    if model_dim != expected_model_dim:
        raise ValueError(
            f"model_dim must equal num_heads * head_dim. "
            f"Got model_dim={model_dim}, num_heads={num_heads}, head_dim={head_dim}."
        )

    x = x.view(batch_size, seq_len, num_heads, head_dim)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


def fused_project_qkv(
    hidden_states: torch.Tensor,
    fused_weights: FusedProjectionWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project hidden states into Q/K/V using one fused matrix multiply.

    Input:
        hidden_states: [B, S, model_dim]

    Output:
        q, k, v: each [B, H, S, D]
    """
    qkv = hidden_states @ fused_weights.W_qkv
    q, k, v = torch.split(qkv, fused_weights.model_dim, dim=-1)

    q = reshape_to_heads(q, fused_weights.num_heads, fused_weights.head_dim)
    k = reshape_to_heads(k, fused_weights.num_heads, fused_weights.head_dim)
    v = reshape_to_heads(v, fused_weights.num_heads, fused_weights.head_dim)

    return q, k, v


def optimized_attention_step(
    q: torch.Tensor,
    k_visible_t: torch.Tensor,
    v_visible: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Manual single-query attention path.

    Inputs:
        q:           [B, H, 1, D]
        k_visible_t: [B, H, D, T]
        v_visible:   [B, H, T, D]

    Output:
        out:         [B, H, 1, D]
    """
    scale = 1.0 / math.sqrt(head_dim)

    # [B, H, 1, T]
    scores = torch.matmul(q, k_visible_t) * scale

    # Softmax in fp32 for better numerical stability, then cast back.
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # [B, H, 1, D]
    out = torch.matmul(probs, v_visible)
    return out


def optimized_prefill_kv_cache(
    hidden_states: torch.Tensor,
    fused_weights: FusedProjectionWeights,
    total_capacity: int,
) -> OptimizedKVCache:
    """
    Prefill optimized KV cache with prompt tokens.

    Input:
        hidden_states: [B, prompt_len, model_dim]
    """
    batch_size = hidden_states.shape[0]

    cache = OptimizedKVCache(
        batch=batch_size,
        heads=fused_weights.num_heads,
        head_dim=fused_weights.head_dim,
        capacity=total_capacity,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    prompt_len = hidden_states.shape[1]
    if prompt_len == 0:
        return cache

    _, k_prompt, v_prompt = fused_project_qkv(hidden_states, fused_weights)
    cache.append_prefix(k_prompt, v_prompt)
    return cache


def optimized_kv_cache_decode_step(
    current_hidden_state: torch.Tensor,
    cache: OptimizedKVCache,
    fused_weights: FusedProjectionWeights,
) -> torch.Tensor:
    """
    One decode step using:
      - fused QKV projection
      - decode-friendly K cache layout
      - manual single-query attention

    Input:
        current_hidden_state: [B, 1, model_dim]

    Output:
        out: [B, H, 1, D]
    """
    if current_hidden_state.size(1) != 1:
        raise ValueError("current_hidden_state must have shape [B, 1, model_dim].")

    q_new, k_new, v_new = fused_project_qkv(current_hidden_state, fused_weights)

    # Append current token K/V first, so token can attend to itself too.
    cache.append(k_new, v_new)

    out = optimized_attention_step(
        q=q_new,
        k_visible_t=cache.get_visible_k_t(),
        v_visible=cache.get_visible_v(),
        head_dim=fused_weights.head_dim,
    )
    return out


def run_optimized_kv_cache_decode(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    fused_weights: FusedProjectionWeights,
) -> torch.Tensor:
    """
    Run the full optimized KV-cache decode loop.

    Output shape:
        [B, H, gen_steps, D]
    """
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len:prompt_len + gen_steps, :]

    cache = optimized_prefill_kv_cache(
        hidden_states=prompt_hidden_states,
        fused_weights=fused_weights,
        total_capacity=total_seq_len,
    )

    outputs = []
    for step_idx in range(gen_steps):
        current_hidden_state = generated_hidden_states[:, step_idx:step_idx + 1, :]
        out = optimized_kv_cache_decode_step(
            current_hidden_state=current_hidden_state,
            cache=cache,
            fused_weights=fused_weights,
        )
        outputs.append(out)

    if len(outputs) == 0:
        return all_hidden_states.new_empty(
            batch_size,
            fused_weights.num_heads,
            0,
            fused_weights.head_dim,
        )

    return torch.cat(outputs, dim=2)