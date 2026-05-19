# src/benchmark/stage4b_experiment.py

import math
from typing import Any

import torch

from src.attention.backend_decode import (
    make_compiled_backend_runner,
    make_preallocated_backend_runner,
)
from src.attention.cuda_graph_decode import make_cuda_graph_backend_runner
from src.attention.decode import run_kv_cache_decode, run_naive_decode
from src.attention.optimized_decode import run_optimized_kv_cache_decode
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
    nan_stats,
    safe_speedup,
    write_rows_to_csv,
)


def build_stage4b_row(
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
    compile_mode: str = "reduce-overhead",
    fullgraph: bool = False,
    enable_compile: bool = True,
    enable_cuda_graphs: bool = True,
) -> dict[str, Any]:
    """
    Run one Stage 4B experiment row.

    Stage 4B compares:
    - naive decode
    - KV-cache decode
    - Stage 4A optimized decode
    - Stage 4B-A eager backend path
    - Stage 4B-B compiled backend path
    - Stage 4B-C CUDA Graph backend path if available
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

    backend_eager_runner = make_preallocated_backend_runner(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        device=state.device,
        dtype=state.dtype,
    )

    backend_compiled_runner, backend_compiled_status = make_compiled_backend_runner(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        device=state.device,
        dtype=state.dtype,
        enable_compile=enable_compile,
        compile_mode=compile_mode,
        fullgraph=fullgraph,
    )

    backend_cuda_graph_runner, backend_cuda_graph_status = make_cuda_graph_backend_runner(
        sample_hidden_states=hidden_states,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        enable_cuda_graphs=enable_cuda_graphs,
    )

    backend_cuda_graph_available = backend_cuda_graph_runner is not None

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

        backend_eager_out = backend_eager_runner(hidden_states)
        backend_compiled_out = backend_compiled_runner(hidden_states)

        backend_cuda_graph_out = None
        if backend_cuda_graph_runner is not None:
            backend_cuda_graph_out = backend_cuda_graph_runner(hidden_states).clone()

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

    naive_vs_backend_eager = compare_tensors(
        naive_out,
        backend_eager_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    naive_vs_backend_compiled = compare_tensors(
        naive_out,
        backend_compiled_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    naive_vs_backend_cuda_graph = compare_tensors(
        naive_out,
        backend_cuda_graph_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    backend_eager_vs_optimized = compare_tensors(
        backend_eager_out,
        optimized_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    backend_compiled_vs_backend_eager = compare_tensors(
        backend_compiled_out,
        backend_eager_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    backend_cuda_graph_vs_backend_compiled = compare_tensors(
        backend_cuda_graph_out,
        backend_compiled_out,
        atol=state.atol,
        rtol=state.rtol,
    )

    all_correct_available_paths = (
        naive_vs_cache["allclose"]
        and naive_vs_optimized["allclose"]
        and cache_vs_optimized["allclose"]
        and naive_vs_backend_eager["allclose"]
        and naive_vs_backend_compiled["allclose"]
        and naive_vs_backend_cuda_graph["allclose"]
        and backend_eager_vs_optimized["allclose"]
        and backend_compiled_vs_backend_eager["allclose"]
        and backend_cuda_graph_vs_backend_compiled["allclose"]
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

    backend_eager_total_stats = benchmark_callable(
        fn=lambda: backend_eager_runner(hidden_states),
        warmup=warmup,
        iters=iters,
        device=state.device,
    )

    backend_compiled_total_stats = benchmark_callable(
        fn=lambda: backend_compiled_runner(hidden_states),
        warmup=warmup,
        iters=iters,
        device=state.device,
    )

    if backend_cuda_graph_runner is not None:
        backend_cuda_graph_total_stats = benchmark_callable(
            fn=lambda: backend_cuda_graph_runner(hidden_states),
            warmup=warmup,
            iters=iters,
            device=state.device,
        )
    else:
        backend_cuda_graph_total_stats = nan_stats()

    naive_amortized_per_step_mean_ms = (
        naive_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    cache_amortized_per_step_mean_ms = (
        cache_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    optimized_amortized_per_step_mean_ms = (
        optimized_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    backend_eager_amortized_per_step_mean_ms = (
        backend_eager_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    backend_compiled_amortized_per_step_mean_ms = (
        backend_compiled_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    backend_cuda_graph_amortized_per_step_mean_ms = (
        backend_cuda_graph_total_stats["mean_ms"] / gen_steps
        if gen_steps > 0 and not math.isnan(backend_cuda_graph_total_stats["mean_ms"])
        else float("nan")
    )

    backend_kv_cache_capacity_bytes = standard_kv_cache_capacity_bytes(state)

    backend_output_buffer_bytes = (
        batch
        * heads
        * gen_steps
        * head_dim
        * hidden_states.element_size()
    )

    row: dict[str, Any] = {
        "stage": "stage4b",
        "stage4a_path_name": "fused_qkv_manual_single_query",
        "stage4b_eager_path_name": "tensor_only_preallocated_eager",
        "stage4b_compiled_path_name": "tensor_only_preallocated_compile",
        "stage4b_cuda_graph_path_name": "tensor_only_preallocated_cuda_graph",

        **build_common_config_row(state),

        "backend_compiled_status": backend_compiled_status,
        "backend_compile_mode": compile_mode,
        "backend_compile_fullgraph": fullgraph,
        "backend_cuda_graph_status": backend_cuda_graph_status,
        "backend_cuda_graph_available": backend_cuda_graph_available,

        "naive_checksum": checksum(naive_out),
        "cache_checksum": checksum(cache_out),
        "optimized_checksum": checksum(optimized_out),
        "backend_eager_checksum": checksum(backend_eager_out),
        "backend_compiled_checksum": checksum(backend_compiled_out),
        "backend_cuda_graph_checksum": checksum(backend_cuda_graph_out),

        "naive_vs_cache_allclose": naive_vs_cache["allclose"],
        "naive_vs_cache_max_abs_diff": naive_vs_cache["max_abs_diff"],
        "naive_vs_cache_mean_abs_diff": naive_vs_cache["mean_abs_diff"],

        "naive_vs_optimized_allclose": naive_vs_optimized["allclose"],
        "naive_vs_optimized_max_abs_diff": naive_vs_optimized["max_abs_diff"],
        "naive_vs_optimized_mean_abs_diff": naive_vs_optimized["mean_abs_diff"],

        "cache_vs_optimized_allclose": cache_vs_optimized["allclose"],
        "cache_vs_optimized_max_abs_diff": cache_vs_optimized["max_abs_diff"],
        "cache_vs_optimized_mean_abs_diff": cache_vs_optimized["mean_abs_diff"],

        "naive_vs_backend_eager_allclose": naive_vs_backend_eager["allclose"],
        "naive_vs_backend_eager_max_abs_diff": naive_vs_backend_eager["max_abs_diff"],
        "naive_vs_backend_eager_mean_abs_diff": naive_vs_backend_eager["mean_abs_diff"],

        "naive_vs_backend_compiled_allclose": naive_vs_backend_compiled["allclose"],
        "naive_vs_backend_compiled_max_abs_diff": naive_vs_backend_compiled[
            "max_abs_diff"
        ],
        "naive_vs_backend_compiled_mean_abs_diff": naive_vs_backend_compiled[
            "mean_abs_diff"
        ],

        "naive_vs_backend_cuda_graph_allclose": naive_vs_backend_cuda_graph["allclose"],
        "naive_vs_backend_cuda_graph_max_abs_diff": naive_vs_backend_cuda_graph[
            "max_abs_diff"
        ],
        "naive_vs_backend_cuda_graph_mean_abs_diff": naive_vs_backend_cuda_graph[
            "mean_abs_diff"
        ],

        "backend_eager_vs_optimized_allclose": backend_eager_vs_optimized["allclose"],
        "backend_eager_vs_optimized_max_abs_diff": backend_eager_vs_optimized[
            "max_abs_diff"
        ],
        "backend_eager_vs_optimized_mean_abs_diff": backend_eager_vs_optimized[
            "mean_abs_diff"
        ],

        "backend_compiled_vs_backend_eager_allclose": backend_compiled_vs_backend_eager[
            "allclose"
        ],
        "backend_compiled_vs_backend_eager_max_abs_diff": (
            backend_compiled_vs_backend_eager["max_abs_diff"]
        ),
        "backend_compiled_vs_backend_eager_mean_abs_diff": (
            backend_compiled_vs_backend_eager["mean_abs_diff"]
        ),

        "backend_cuda_graph_vs_backend_compiled_allclose": (
            backend_cuda_graph_vs_backend_compiled["allclose"]
        ),
        "backend_cuda_graph_vs_backend_compiled_max_abs_diff": (
            backend_cuda_graph_vs_backend_compiled["max_abs_diff"]
        ),
        "backend_cuda_graph_vs_backend_compiled_mean_abs_diff": (
            backend_cuda_graph_vs_backend_compiled["mean_abs_diff"]
        ),

        "all_correct_available_paths": all_correct_available_paths,

        **build_standard_attention_memory_row(state),

        "backend_kv_cache_capacity_bytes": backend_kv_cache_capacity_bytes,
        "backend_kv_cache_capacity_mib": bytes_to_mib(backend_kv_cache_capacity_bytes),

        "backend_output_buffer_bytes": backend_output_buffer_bytes,
        "backend_output_buffer_mib": bytes_to_mib(backend_output_buffer_bytes),

        **flatten_stats("naive_full_total", naive_total_stats),
        **flatten_stats("cache_full_total", cache_total_stats),
        **flatten_stats("optimized_full_total", optimized_total_stats),
        **flatten_stats("backend_eager_full_total", backend_eager_total_stats),
        **flatten_stats("backend_compiled_full_total", backend_compiled_total_stats),
        **flatten_stats("backend_cuda_graph_full_total", backend_cuda_graph_total_stats),

        "naive_amortized_per_step_mean_ms": naive_amortized_per_step_mean_ms,
        "cache_amortized_per_step_mean_ms": cache_amortized_per_step_mean_ms,
        "optimized_amortized_per_step_mean_ms": optimized_amortized_per_step_mean_ms,
        "backend_eager_amortized_per_step_mean_ms": (
            backend_eager_amortized_per_step_mean_ms
        ),
        "backend_compiled_amortized_per_step_mean_ms": (
            backend_compiled_amortized_per_step_mean_ms
        ),
        "backend_cuda_graph_amortized_per_step_mean_ms": (
            backend_cuda_graph_amortized_per_step_mean_ms
        ),

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

        "backend_eager_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"],
            backend_eager_total_stats["mean_ms"],
        ),
        "backend_compiled_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"],
            backend_compiled_total_stats["mean_ms"],
        ),
        "backend_cuda_graph_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"],
            backend_cuda_graph_total_stats["mean_ms"],
        ),
        "backend_compiled_vs_backend_eager_full_speedup": safe_speedup(
            backend_eager_total_stats["mean_ms"],
            backend_compiled_total_stats["mean_ms"],
        ),
        "backend_cuda_graph_vs_backend_compiled_full_speedup": safe_speedup(
            backend_compiled_total_stats["mean_ms"],
            backend_cuda_graph_total_stats["mean_ms"],
        ),
    }

    return row