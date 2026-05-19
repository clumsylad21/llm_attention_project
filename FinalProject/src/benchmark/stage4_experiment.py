# src/benchmark/stage4_experiment.py

from typing import Any

import torch

from src.attention.decode import (
    create_projection_weights,
    run_kv_cache_decode,
    run_naive_decode,
)
from src.attention.optimized_decode import (
    build_fused_projection_weights_from_separate,
    optimized_prefill_kv_cache,
    run_optimized_kv_cache_decode,
)
from src.benchmark.experiment_utils import (
    benchmark_callable,
    bytes_to_mib,
    checksum,
    compare_tensors,
    default_tolerances,
    flatten_stats,
    resolve_device,
    resolve_dtype,
    safe_speedup,
    set_seed,
    tensor_bytes,
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
    """
    device = resolve_device(device_requested)
    dtype = resolve_dtype(dtype_name)

    model_dim = heads * head_dim
    total_seq_len = prompt_len + gen_steps

    set_seed(seed)

    hidden_states = torch.randn(
        batch,
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
        heads=heads,
        head_dim=head_dim,
    )

    atol, rtol = default_tolerances(dtype)

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

    naive_vs_cache = compare_tensors(naive_out, cache_out, atol=atol, rtol=rtol)
    naive_vs_optimized = compare_tensors(naive_out, optimized_out, atol=atol, rtol=rtol)
    cache_vs_optimized = compare_tensors(cache_out, optimized_out, atol=atol, rtol=rtol)

    hidden_states_bytes = tensor_bytes(hidden_states)

    separate_weights_bytes = (
        tensor_bytes(separate_weights.w_q)
        + tensor_bytes(separate_weights.w_k)
        + tensor_bytes(separate_weights.w_v)
    )

    fused_weights_bytes = tensor_bytes(fused_weights.W_qkv)

    element_size = hidden_states.element_size()

    standard_kv_cache_capacity_bytes = (
        2 * batch * heads * total_seq_len * head_dim * element_size
    )

    optimized_kv_cache_capacity_bytes = standard_kv_cache_capacity_bytes

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
        device=device,
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
        device=device,
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
        device=device,
    )

    optimized_prefill_stats = benchmark_callable(
        fn=lambda: optimized_prefill_kv_cache(
            hidden_states=hidden_states[:, :prompt_len, :],
            fused_weights=fused_weights,
            total_capacity=total_seq_len,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
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

    row: dict[str, Any] = {
        "stage": "stage4",
        "optimized_path_name": "fused_qkv_manual_single_query",
        "optimization_target": "reduce_constant_overhead_inside_kv_cache_decode",
        "device_requested": device_requested,
        "resolved_device": str(device),
        "dtype_name": dtype_name,
        "resolved_dtype": str(dtype),
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "model_dim": model_dim,
        "prompt_len": prompt_len,
        "gen_steps": gen_steps,
        "total_seq_len": total_seq_len,
        "warmup": warmup,
        "iters": iters,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
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
        "hidden_states_bytes": hidden_states_bytes,
        "hidden_states_mib": bytes_to_mib(hidden_states_bytes),
        "separate_weights_bytes": separate_weights_bytes,
        "separate_weights_mib": bytes_to_mib(separate_weights_bytes),
        "fused_weights_bytes": fused_weights_bytes,
        "fused_weights_mib": bytes_to_mib(fused_weights_bytes),
        "standard_kv_cache_capacity_bytes": standard_kv_cache_capacity_bytes,
        "standard_kv_cache_capacity_mib": bytes_to_mib(standard_kv_cache_capacity_bytes),
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