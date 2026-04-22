# src/attention/stage6_custom_decode.py

from __future__ import annotations

import torch

from src.attention.optimized_decode import (
    FusedProjectionWeights,
    fused_project_qkv,
    optimized_prefill_kv_cache,
)
from src.attention.stage6_cuda_extension import stage6_single_query_attention


def stage6_custom_attention_step(
    q: torch.Tensor,
    k_visible_t: torch.Tensor,
    v_visible: torch.Tensor,
) -> torch.Tensor:
    """
    Thin Python wrapper around the custom CUDA extension.

    This function is intentionally small:
    Stage 6 custom work is only the attention core.
    Everything else remains in the existing project structure.
    """
    return stage6_single_query_attention(
        q=q,
        k_visible_t=k_visible_t,
        v_visible=v_visible,
    )


def run_stage6_custom_decode(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    fused_weights: FusedProjectionWeights,
) -> torch.Tensor:
    """
    Run the full decode loop using:
    - Stage 4A fused QKV projection
    - Stage 4A decode-friendly KV cache layout
    - Stage 6 custom CUDA attention core

    Output:
        [B, H, gen_steps, D]
    """
    if all_hidden_states.device.type != "cuda":
        raise ValueError("Stage 6 custom decode requires CUDA tensors.")

    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len : prompt_len + gen_steps, :]

    cache = optimized_prefill_kv_cache(
        hidden_states=prompt_hidden_states,
        fused_weights=fused_weights,
        total_capacity=total_seq_len,
    )

    outputs = []
    for step_idx in range(gen_steps):
        current_hidden_state = generated_hidden_states[:, step_idx : step_idx + 1, :]

        q_new, k_new, v_new = fused_project_qkv(current_hidden_state, fused_weights)

        # Just like the existing decode paths, append current token first
        # so the token can attend to itself too.
        cache.append(k_new, v_new)

        out = stage6_custom_attention_step(
            q=q_new,
            k_visible_t=cache.get_visible_k_t(),
            v_visible=cache.get_visible_v(),
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
