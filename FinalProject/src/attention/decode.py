# src/attention/decode.py

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.attention.kv_cache import KVCache


@dataclass
class ProjectionWeights:
    """
    Synthetic decoder projection weights.

    Each weight is [model_dim, model_dim].

    We keep them separate to match the standard idea of
    Wq, Wk, Wv in transformer attention.
    """
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor


def create_projection_weights(
    model_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ProjectionWeights:
    """
    Create synthetic Q/K/V projection weights.

    We scale by 1/sqrt(model_dim) so values do not blow up too much.
    """
    scale = 1.0 / math.sqrt(model_dim)

    w_q = torch.randn(model_dim, model_dim, device=device, dtype=dtype) * scale
    w_k = torch.randn(model_dim, model_dim, device=device, dtype=dtype) * scale
    w_v = torch.randn(model_dim, model_dim, device=device, dtype=dtype) * scale

    return ProjectionWeights(w_q=w_q, w_k=w_k, w_v=w_v)


def reshape_to_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Convert hidden states from [B, S, model_dim] to [B, H, S, D].

    model_dim must be equal to num_heads * head_dim.
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


def project_qkv(
    hidden_states: torch.Tensor,
    weights: ProjectionWeights,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project hidden states into Q, K, V and reshape to multi-head format.

    Input:
        hidden_states: [B, S, model_dim]

    Output:
        q, k, v: each [B, H, S, D]
    """
    q = hidden_states @ weights.w_q
    k = hidden_states @ weights.w_k
    v = hidden_states @ weights.w_v

    q = reshape_to_heads(q, num_heads, head_dim)
    k = reshape_to_heads(k, num_heads, head_dim)
    v = reshape_to_heads(v, num_heads, head_dim)

    return q, k, v


def attention_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Run scaled dot-product attention for a single decode step.

    Expected shapes:
        q: [B, H, 1, D]
        k: [B, H, S_visible, D]
        v: [B, H, S_visible, D]

    We use is_causal=False here because k and v already contain only the
    visible prefix. There are no future tokens in k/v to mask away.
    """
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
    )


def naive_decode_step(
    all_hidden_states: torch.Tensor,
    weights: ProjectionWeights,
    num_heads: int,
    head_dim: int,
    visible_len: int,
) -> torch.Tensor:
    """
    Naive decode step.

    At every step we recompute K and V for the full visible prefix.

    Example:
        visible_len = prompt_len + step_idx + 1

    That means the current token is already included inside the visible prefix,
    so the current query can attend to all prior tokens plus itself.
    """
    if visible_len < 1 or visible_len > all_hidden_states.size(1):
        raise ValueError(
            f"visible_len must be in [1, total_seq_len]. Got {visible_len}."
        )

    visible_states = all_hidden_states[:, :visible_len, :]  # [B, visible_len, model_dim]

    q_all, k_all, v_all = project_qkv(
        visible_states,
        weights,
        num_heads,
        head_dim,
    )

    # Use only the newest token as the decode query.
    q_current = q_all[:, :, -1:, :]  # [B, H, 1, D]

    out = attention_step(q_current, k_all, v_all)
    return out


def prefill_kv_cache(
    prompt_hidden_states: torch.Tensor,
    weights: ProjectionWeights,
    num_heads: int,
    head_dim: int,
    cache: KVCache,
) -> None:
    """
    Fill the KV cache using the prompt tokens.

    This is the 'prefill' part of decode-style inference:
    prompt tokens are processed once, then future decode steps reuse them.
    """
    _, k_prompt, v_prompt = project_qkv(
        prompt_hidden_states,
        weights,
        num_heads,
        head_dim,
    )
    cache.append(k_prompt, v_prompt)


def kv_cache_decode_step(
    current_hidden_state: torch.Tensor,
    weights: ProjectionWeights,
    num_heads: int,
    head_dim: int,
    cache: KVCache,
) -> torch.Tensor:
    """
    KV-cache decode step.

    Input:
        current_hidden_state: [B, 1, model_dim]

    Workflow:
        1. Project current token into q, k, v
        2. Append new k/v to the cache
        3. Run attention using q against all cached k/v
    """
    if current_hidden_state.size(1) != 1:
        raise ValueError("current_hidden_state must have shape [B, 1, model_dim].")

    q_new, k_new, v_new = project_qkv(
        current_hidden_state,
        weights,
        num_heads,
        head_dim,
    )

    # Append new token K/V first, so the token can attend to itself too.
    cache.append(k_new, v_new)

    k_visible, v_visible = cache.get_kv()
    out = attention_step(q_new, k_visible, v_visible)
    return out


def run_naive_decode(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    weights: ProjectionWeights,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Run the full naive decode loop and return outputs from every generated step.

    Output shape:
        [B, H, gen_steps, D]
    """
    outputs = []

    for step_idx in range(gen_steps):
        visible_len = prompt_len + step_idx + 1
        out = naive_decode_step(
            all_hidden_states,
            weights,
            num_heads,
            head_dim,
            visible_len,
        )
        outputs.append(out)

    return torch.cat(outputs, dim=2)


def run_kv_cache_decode(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    weights: ProjectionWeights,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Run the full KV-cache decode loop and return outputs from every generated step.

    Output shape:
        [B, H, gen_steps, D]
    """
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len:prompt_len + gen_steps, :]

    cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=total_seq_len,
        head_dim=head_dim,
        device=all_hidden_states.device,
        dtype=all_hidden_states.dtype,
    )

    prefill_kv_cache(
        prompt_hidden_states,
        weights,
        num_heads,
        head_dim,
        cache,
    )

    outputs = []
    for step_idx in range(gen_steps):
        current_hidden_state = generated_hidden_states[:, step_idx:step_idx + 1, :]
        out = kv_cache_decode_step(
            current_hidden_state,
            weights,
            num_heads,
            head_dim,
            cache,
        )
        outputs.append(out)

    return torch.cat(outputs, dim=2)