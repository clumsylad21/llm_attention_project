# src/attention/stage6_custom_decode.py
from __future__ import annotations

import torch

from src.attention.optimized_decode import (
    FusedProjectionWeights,
    fused_project_qkv,
    optimized_prefill_kv_cache,
)
from src.attention.stage6_cuda_extension import get_stage6_extension


def stage6_custom_attention_step(
    q: torch.Tensor,
    k_visible_t: torch.Tensor,
    v_visible: torch.Tensor,
    module=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Thin Python wrapper around the custom CUDA extension.

    This stays intentionally small:
    Stage 6 custom work is only the attention core.
    Everything else remains in the existing project structure.
    """
    if module is None:
        module, status = get_stage6_extension(
            device=q.device,
            dtype=q.dtype,
            enable_stage6=True,
        )
        if module is None:
            raise RuntimeError(f"Stage 6 extension unavailable: {status}")

    q_contig = q.contiguous()
    k_contig = k_visible_t.contiguous()
    v_contig = v_visible.contiguous()

    if out is None:
        return module.forward(q_contig, k_contig, v_contig)

    if not out.is_contiguous():
        raise ValueError("Stage 6 preallocated output buffer must be contiguous.")

    module.forward_out(q_contig, k_contig, v_contig, out)
    return out


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

    if gen_steps == 0:
        return all_hidden_states.new_empty(
            batch_size,
            fused_weights.num_heads,
            0,
            fused_weights.head_dim,
        )

    module, status = get_stage6_extension(
        device=all_hidden_states.device,
        dtype=all_hidden_states.dtype,
        enable_stage6=True,
    )
    if module is None:
        raise RuntimeError(f"Stage 6 extension unavailable: {status}")

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len : prompt_len + gen_steps, :]

    cache = optimized_prefill_kv_cache(
        hidden_states=prompt_hidden_states,
        fused_weights=fused_weights,
        total_capacity=total_seq_len,
    )

    outputs = all_hidden_states.new_empty(
        batch_size,
        fused_weights.num_heads,
        gen_steps,
        fused_weights.head_dim,
    )

    # Reuse one step-sized output buffer to avoid per-step allocation.
    step_out = all_hidden_states.new_empty(
        batch_size,
        fused_weights.num_heads,
        1,
        fused_weights.head_dim,
    )

    for step_idx in range(gen_steps):
        current_hidden_state = generated_hidden_states[:, step_idx : step_idx + 1, :]

        q_new, k_new, v_new = fused_project_qkv(current_hidden_state, fused_weights)

        # Append current token first so the token can attend to itself too.
        cache.append(k_new, v_new)

        stage6_custom_attention_step(
            q=q_new,
            k_visible_t=cache.get_visible_k_t(),
            v_visible=cache.get_visible_v(),
            module=module,
            out=step_out,
        )

        outputs[:, :, step_idx : step_idx + 1, :].copy_(step_out)

    return outputs
