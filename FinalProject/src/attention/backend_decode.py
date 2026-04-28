# src/attention/backend_decode.py

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from src.attention.optimized_decode import FusedProjectionWeights


@dataclass
class BackendDecodeBuffers:
    """
    Preallocated buffers for the tensor-only backend decode path.

    k_cache_t:
        [B, H, D, total_seq_len]
        K is stored already transposed for decode-friendly matmul.

    v_cache:
        [B, H, total_seq_len, D]

    outputs:
        [B, H, gen_steps, D]
    """
    k_cache_t: torch.Tensor
    v_cache: torch.Tensor
    outputs: torch.Tensor


def allocate_backend_decode_buffers(
    batch: int,
    heads: int,
    head_dim: int,
    total_seq_len: int,
    gen_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BackendDecodeBuffers:
    """
    Allocate all fixed-shape buffers once.

    This is the heart of Stage 4B-A:
    we stop building Python cache objects inside the hot path,
    and instead reuse static tensors.
    """
    k_cache_t = torch.empty(
        batch,
        heads,
        head_dim,
        total_seq_len,
        device=device,
        dtype=dtype,
    )

    v_cache = torch.empty(
        batch,
        heads,
        total_seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    outputs = torch.empty(
        batch,
        heads,
        gen_steps,
        head_dim,
        device=device,
        dtype=dtype,
    )

    return BackendDecodeBuffers(
        k_cache_t=k_cache_t,
        v_cache=v_cache,
        outputs=outputs,
    )


def reshape_to_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Convert:
        [B, S, model_dim]
    into:
        [B, H, S, D]
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One fused QKV projection.

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


def manual_single_query_attention(
    q: torch.Tensor,
    visible_k_t: torch.Tensor,
    visible_v: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Manual single-query attention specialized for decode.

    Inputs:
        q           : [B, H, 1, D]
        visible_k_t : [B, H, D, T]
        visible_v   : [B, H, T, D]

    Output:
        out         : [B, H, 1, D]
    """
    scale = 1.0 / math.sqrt(head_dim)

    # [B, H, 1, T]
    scores = torch.matmul(q, visible_k_t) * scale

    # Softmax in fp32 for a little extra stability.
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # [B, H, 1, D]
    out = torch.matmul(probs, visible_v)
    return out


def run_backend_decode_preallocated(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    fused_weights: FusedProjectionWeights,
    buffers: BackendDecodeBuffers,
) -> torch.Tensor:
    """
    Tensor-only fixed-shape decode path.

    This is Stage 4B-A.

    Important idea:
    - no Python cache object inside the hot path
    - no growing lists
    - no repeated torch.cat for outputs
    - all K/V/output storage is preallocated
    """
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)
    expected_total_seq_len = prompt_len + gen_steps

    if total_seq_len != expected_total_seq_len:
        raise ValueError(
            f"Expected total_seq_len={expected_total_seq_len}, got {total_seq_len}."
        )

    if buffers.k_cache_t.size(0) != batch_size:
        raise ValueError("Buffer batch size does not match input batch size.")

    k_cache_t = buffers.k_cache_t
    v_cache = buffers.v_cache
    outputs = buffers.outputs

    # ------------------------------------------------------------
    # 1) Prefill prompt K/V once
    # ------------------------------------------------------------
    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]

    if prompt_len > 0:
        _, k_prompt, v_prompt = fused_project_qkv(prompt_hidden_states, fused_weights)

        # k_prompt is [B, H, prompt_len, D]
        # store K already transposed as [B, H, D, prompt_len]
        k_cache_t[:, :, :, :prompt_len].copy_(k_prompt.transpose(-2, -1).contiguous())

        # v_prompt already matches [B, H, prompt_len, D]
        v_cache[:, :, :prompt_len, :].copy_(v_prompt.contiguous())

    # ------------------------------------------------------------
    # 2) Decode one token at a time
    # ------------------------------------------------------------
    for step_idx in range(gen_steps):
        token_index = prompt_len + step_idx

        # [B, 1, model_dim]
        current_hidden_state = all_hidden_states[:, token_index : token_index + 1, :]

        # q/k/v each [B, H, 1, D]
        q_new, k_new, v_new = fused_project_qkv(current_hidden_state, fused_weights)

        # Append current token into preallocated cache
        k_cache_t[:, :, :, token_index : token_index + 1].copy_(
            k_new.transpose(-2, -1).contiguous()
        )
        v_cache[:, :, token_index : token_index + 1, :].copy_(v_new.contiguous())

        visible_len = token_index + 1

        visible_k_t = k_cache_t[:, :, :, :visible_len]
        visible_v = v_cache[:, :, :visible_len, :]

        out = manual_single_query_attention(
            q=q_new,
            visible_k_t=visible_k_t,
            visible_v=visible_v,
            head_dim=fused_weights.head_dim,
        )

        outputs[:, :, step_idx : step_idx + 1, :].copy_(out)

    return outputs


class PreallocatedBackendDecodeRunner:
    """
    Reusable eager runner for the Stage 4B-A path.

    Buffers are allocated once in __init__ and reused across calls.
    """

    def __init__(
        self,
        batch: int,
        prompt_len: int,
        gen_steps: int,
        fused_weights: FusedProjectionWeights,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.prompt_len = prompt_len
        self.gen_steps = gen_steps
        self.fused_weights = fused_weights
        self.device = device
        self.dtype = dtype

        total_seq_len = prompt_len + gen_steps

        self.buffers = allocate_backend_decode_buffers(
            batch=batch,
            heads=fused_weights.num_heads,
            head_dim=fused_weights.head_dim,
            total_seq_len=total_seq_len,
            gen_steps=gen_steps,
            device=device,
            dtype=dtype,
        )

    def __call__(self, all_hidden_states: torch.Tensor) -> torch.Tensor:
        return run_backend_decode_preallocated(
            all_hidden_states=all_hidden_states,
            prompt_len=self.prompt_len,
            gen_steps=self.gen_steps,
            fused_weights=self.fused_weights,
            buffers=self.buffers,
        )


def make_preallocated_backend_runner(
    batch: int,
    prompt_len: int,
    gen_steps: int,
    fused_weights: FusedProjectionWeights,
    device: torch.device,
    dtype: torch.dtype,
) -> PreallocatedBackendDecodeRunner:
    """
    Build the Stage 4B-A eager runner.
    """
    return PreallocatedBackendDecodeRunner(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        device=device,
        dtype=dtype,
    )


def make_compiled_backend_runner(
    batch: int,
    prompt_len: int,
    gen_steps: int,
    fused_weights: FusedProjectionWeights,
    device: torch.device,
    dtype: torch.dtype,
    enable_compile: bool = True,
    compile_mode: str = "reduce-overhead",
    fullgraph: bool = False,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """
    Build the Stage 4B-B runner.

    Returns:
        runner, status_string
    """
    eager_runner = make_preallocated_backend_runner(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        device=device,
        dtype=dtype,
    )

    if not enable_compile:
        return eager_runner, "compile_disabled"

    if not hasattr(torch, "compile"):
        return eager_runner, "torch_compile_unavailable"

    def eager_fn(all_hidden_states: torch.Tensor) -> torch.Tensor:
        return eager_runner(all_hidden_states)

    try:
        compiled_fn = torch.compile(
            eager_fn,
            mode=compile_mode,
            fullgraph=fullgraph,
        )
        return compiled_fn, f"compiled_{compile_mode}"
    except Exception as exc:
        # Graceful fallback keeps the project runnable.
        return eager_runner, f"compile_fallback_{type(exc).__name__}"