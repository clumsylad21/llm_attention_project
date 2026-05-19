# src/benchmark/stage4_experiment.py

from typing import Any

import torch

from src.attention.decode import run_kv_cache_decode, run_naive_decode
from src.attention.optimized_decode import (
    optimized_prefill_kv_cache,
    run_optimized_kv_cache_decode,
)
from src.benchmark.attention_experiment_common import (
    AttentionExperimentConfig,
    build_common_config_row,
    build_standard_attention_memory_row,
    prepare_attention_experiment,
    standard_kv_cache_capacity_bytes,
)
from src.benchmark.experiment_utils import (
    benchmark_callable,
    bytes_to_mib,
    checksum,
    compare_tensors,
    flatten_stats,
    safe_speedup,
    write_rows_to_csv,
)


def build_stage4_row(
    device_requested: str,
    dtype_name: str,
    batch: int,
    heads: int,
    head_dim: int,
    prompt_len: int,
    gen_steps: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    """
    Run one Stage 4 experiment and return a flat row dictionary.

    Stage 4 compares:
    - naive decode
    - KV-cache decode
    - optimized fused-QKV KV-cache decode
    """
    config = AttentionExperimentConfig(
        device_requested=device_requested,
        dtype_name=dtype_name,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        warmup=warmup,
        iters=iters,
        seed=seed,
    )

    state = prepare_attention_experiment(config)

    hidden_states = state.hidden_states
    separate_weights = state.separate_weights
    fused_weights = state.fused_weights

    with torch.no_grad():
        naive_out = run_naive_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            separate_weights,
            heads,
            head_dim,
        )

        cache_out = run_kv_cache_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            separate_weights,
            heads,
            head_dim,
        )

        optimized_out = run_optimized_kv_cache_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            fused_weights,
        )

    naive_vs_cache = compare_tensors(
        naive_out,
        cache_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    naive_vs_optimized = compare_tensors(
        naive_out,
        optimized_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    cache_vs_optimized = compare_tensors(
        cache_out,
        optimized_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    naive_total_stats = benchmark_callable(
        fn=lambda: run_naive_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            separate_weights,
            heads,
            head_dim,
        ),
        warmup=warmup,
        iters=iters,
        device=state.device,
    )

    cache_total_stats = benchmark_callable(
        fn=lambda: run_kv_cache_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            separate_weights,
            heads,
            head_dim,
        ),
        warmup=warmup,
        iters=iters,
        device=state.device,
    )

    optimized_total_stats = benchmark_callable(
        fn=lambda: run_optimized_kv_cache_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            fused_weights,
        ),
        warmup=warmup,
        iters=iters,
        device=state.device,
    )

    optimized_prefill_stats = benchmark_callable(
        fn=lambda: optimized_prefill_kv_cache(
            hidden_states=hidden_states[:, :prompt_len, :],
            fused_weights=fused_weights,
            total_capacity=state.total_seq_len,
        ),
        warmup=warmup,
        iters=iters,
        device=state.device,
    )

    naive_amortized_per_step_mean_ms = (
        naive_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    cache_amortized_per_step_mean_ms = (
        cache_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    optimized_amortized_per_step_mean_ms = (
        optimized_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    optimized_prefill_fraction = (
        optimized_prefill_stats["mean_ms"] / optimized_total_stats["mean_ms"]
        if optimized_total_stats["mean_ms"] > 0.0
        else 0.0
    )

    optimized_kv_cache_capacity_bytes = standard_kv_cache_capacity_bytes(state)

    row: dict[str, Any] = {
        "stage": "stage4",
        "optimized_path_name": "fused_qkv_manual_single_query",
        "optimization_target": "reduce_constant_overhead_inside_kv_cache_decode",

        **build_common_config_row(state),

        "naive_checksum": checksum(naive_out),
        "cache_checksum": checksum(cache_out),
        "optimized_checksum": checksum(optimized_out),

        "naive_vs_cache_allclose": naive_vs_cache["allclose"],
        "naive_vs_cache_max_abs_diff": naive_vs_cache["max_abs_diff"],
        "naive_vs_cache_mean_abs_diff": naive_vs_cache["mean_abs_diff"],

        "naive_vs_optimized_allclose": naive_vs_optimized["allclose"],
        "naive_vs_optimized_max_abs_diff": naive_vs_optimized["max_abs_diff"],
        "naive_vs_optimized_mean_abs_diff": naive_vs_optimized["mean_abs_diff"],

        "cache_vs_optimized_allclose": cache_vs_optimized["allclose"],
        "cache_vs_optimized_max_abs_diff": cache_vs_optimized["max_abs_diff"],
        "cache_vs_optimized_mean_abs_diff": cache_vs_optimized["mean_abs_diff"],

        **build_standard_attention_memory_row(state),

        "optimized_kv_cache_capacity_bytes": optimized_kv_cache_capacity_bytes,
        "optimized_kv_cache_capacity_mib": bytes_to_mib(
            optimized_kv_cache_capacity_bytes
        ),

        **flatten_stats("naive_full_total", naive_total_stats),
        **flatten_stats("cache_full_total", cache_total_stats),
        **flatten_stats("optimized_full_total", optimized_total_stats),
        **flatten_stats("optimized_prefill_only", optimized_prefill_stats),

        "naive_amortized_per_step_mean_ms": naive_amortized_per_step_mean_ms,
        "cache_amortized_per_step_mean_ms": cache_amortized_per_step_mean_ms,
        "optimized_amortized_per_step_mean_ms": optimized_amortized_per_step_mean_ms,
        "optimized_prefill_fraction": optimized_prefill_fraction,

        "cache_vs_naive_full_speedup": safe_speedup(
            naive_total_stats["mean_ms"],
            cache_total_stats["mean_ms"],
        ),
        "optimized_vs_naive_full_speedup": safe_speedup(
            naive_total_stats["mean_ms"],
            optimized_total_stats["mean_ms"],
        ),
        "optimized_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"],
            optimized_total_stats["mean_ms"],
        ),
    }

    return row