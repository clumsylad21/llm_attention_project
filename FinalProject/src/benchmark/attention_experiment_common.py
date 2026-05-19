# src/benchmark/attention_experiment_common.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.attention.decode import create_projection_weights
from src.attention.optimized_decode import build_fused_projection_weights_from_separate
from src.benchmark.experiment_utils import (
    bytes_to_mib,
    default_tolerances,
    resolve_device,
    resolve_dtype,
    set_seed,
    tensor_bytes,
)


@dataclass(frozen=True)
class AttentionExperimentConfig:
    device_requested: str
    dtype_name: str
    batch: int
    heads: int
    head_dim: int
    prompt_len: int
    gen_steps: int
    warmup: int
    iters: int
    seed: int


@dataclass
class AttentionExperimentState:
    config: AttentionExperimentConfig
    device: torch.device
    dtype: torch.dtype
    model_dim: int
    total_seq_len: int
    hidden_states: torch.Tensor
    separate_weights: Any
    fused_weights: Any
    atol: float
    rtol: float


def prepare_attention_experiment(
    config: AttentionExperimentConfig,
) -> AttentionExperimentState:
    """
    Build the shared synthetic attention experiment state.

    This centralizes the setup that Stage 4, Stage 4B, and Stage 6 all need:
    device/dtype resolution, synthetic hidden states, projection weights,
    fused QKV weights, and dtype-dependent tolerances.
    """
    device = resolve_device(config.device_requested)
    dtype = resolve_dtype(config.dtype_name)

    model_dim = config.heads * config.head_dim
    total_seq_len = config.prompt_len + config.gen_steps

    set_seed(config.seed)

    hidden_states = torch.randn(
        config.batch,
        total_seq_len,
        model_dim,
        device=device,
        dtype=dtype,
    )

    separate_weights = create_projection_weights(
        model_dim=model_dim,
        device=device,
        dtype=dtype,
    )

    fused_weights = build_fused_projection_weights_from_separate(
        weights=separate_weights,
        heads=config.heads,
        head_dim=config.head_dim,
    )

    atol, rtol = default_tolerances(dtype)

    return AttentionExperimentState(
        config=config,
        device=device,
        dtype=dtype,
        model_dim=model_dim,
        total_seq_len=total_seq_len,
        hidden_states=hidden_states,
        separate_weights=separate_weights,
        fused_weights=fused_weights,
        atol=atol,
        rtol=rtol,
    )


def build_common_config_row(state: AttentionExperimentState) -> dict[str, Any]:
    """
    Common CSV/config fields shared by attention benchmark stages.
    """
    config = state.config

    return {
        "device_requested": config.device_requested,
        "resolved_device": str(state.device),
        "dtype_name": config.dtype_name,
        "resolved_dtype": str(state.dtype),
        "batch": config.batch,
        "heads": config.heads,
        "head_dim": config.head_dim,
        "model_dim": state.model_dim,
        "prompt_len": config.prompt_len,
        "gen_steps": config.gen_steps,
        "total_seq_len": state.total_seq_len,
        "warmup": config.warmup,
        "iters": config.iters,
        "seed": config.seed,
        "atol": state.atol,
        "rtol": state.rtol,
    }


def standard_kv_cache_capacity_bytes(state: AttentionExperimentState) -> int:
    """
    Capacity for K and V cache:
    2 * batch * heads * total_seq_len * head_dim * element_size.
    """
    config = state.config
    element_size = state.hidden_states.element_size()

    return (
        2
        * config.batch
        * config.heads
        * state.total_seq_len
        * config.head_dim
        * element_size
    )


def build_standard_attention_memory_row(
    state: AttentionExperimentState,
) -> dict[str, Any]:
    """
    Common memory bookkeeping for attention benchmark stages.
    """
    hidden_states_bytes = tensor_bytes(state.hidden_states)

    separate_weights_bytes = (
        tensor_bytes(state.separate_weights.w_q)
        + tensor_bytes(state.separate_weights.w_k)
        + tensor_bytes(state.separate_weights.w_v)
    )

    fused_weights_bytes = tensor_bytes(state.fused_weights.W_qkv)

    standard_cache_bytes = standard_kv_cache_capacity_bytes(state)

    return {
        "hidden_states_bytes": hidden_states_bytes,
        "hidden_states_mib": bytes_to_mib(hidden_states_bytes),
        "separate_weights_bytes": separate_weights_bytes,
        "separate_weights_mib": bytes_to_mib(separate_weights_bytes),
        "fused_weights_bytes": fused_weights_bytes,
        "fused_weights_mib": bytes_to_mib(fused_weights_bytes),
        "standard_kv_cache_capacity_bytes": standard_cache_bytes,
        "standard_kv_cache_capacity_mib": bytes_to_mib(standard_cache_bytes),
    }
