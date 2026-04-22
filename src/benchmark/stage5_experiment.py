# src/benchmark/stage5_experiment.py

import math
from typing import Any

from src.benchmark.stage4b_experiment import (
    build_stage4b_row,
    safe_speedup,
    write_rows_to_csv,
)


def _best_final_path_name_and_latency(row: dict[str, Any]) -> tuple[str, float]:
    """
    Pick the fastest path among the final Stage 5 comparison set.

    Final Stage 5 path set:
    - naive
    - cache
    - Stage 4A
    - compiled
    - CUDA Graph (GPU only if available)
    """
    candidates = [
        ("naive", row["naive_full_total_mean_ms"]),
        ("cache", row["cache_full_total_mean_ms"]),
        ("stage4a", row["stage4a_full_total_mean_ms"]),
        ("compiled", row["compiled_full_total_mean_ms"]),
    ]

    if row["backend_cuda_graph_available"]:
        candidates.append(("cuda_graph", row["cuda_graph_full_total_mean_ms"]))

    valid_candidates = []
    for name, value in candidates:
        if not math.isnan(value):
            valid_candidates.append((name, value))

    if len(valid_candidates) == 0:
        return "unknown", float("nan")

    best_name, best_latency = min(valid_candidates, key=lambda x: x[1])
    return best_name, best_latency


def _final_path_keys(resolved_device: str, backend_cuda_graph_available: bool) -> str:
    """
    Store final path keys in one CSV field for readability/debugging.
    """
    keys = ["naive", "cache", "stage4a", "compiled"]
    if resolved_device == "cuda" and backend_cuda_graph_available:
        keys.append("cuda_graph")
    return ",".join(keys)


def build_stage5_row(
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
    Build one final Stage 5 row.

    Design choice:
    - Reuse the already-working Stage 4B experiment builder
    - Do NOT reimplement all decode logic again
    - Curate Stage 4B into the final Stage 5 comparison set

    Final Stage 5 paths:
    - naive decode
    - KV-cache decode
    - Stage 4A optimized decode
    - 4B-B compiled backend decode
    - 4B-C CUDA Graph decode (GPU only if available)

    Excluded from final main comparison:
    - 4B-A eager backend decode
    """

    raw = build_stage4b_row(
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
        compile_mode=compile_mode,
        fullgraph=fullgraph,
        enable_compile=enable_compile,
        enable_cuda_graphs=enable_cuda_graphs,
    )

    backend_cuda_graph_available = bool(raw["backend_cuda_graph_available"])

    # ------------------------------------------------------------
    # Stage 5 correctness:
    # only check the final included paths
    # ------------------------------------------------------------
    all_correct_final_paths = (
        bool(raw["naive_vs_cache_allclose"])
        and bool(raw["naive_vs_optimized_allclose"])
        and bool(raw["cache_vs_optimized_allclose"])
        and bool(raw["naive_vs_backend_compiled_allclose"])
        and (
            (not backend_cuda_graph_available)
            or bool(raw["naive_vs_backend_cuda_graph_allclose"])
        )
    )

    best_final_path_name, best_final_path_mean_ms = _best_final_path_name_and_latency(
        {
            "naive_full_total_mean_ms": raw["naive_full_total_mean_ms"],
            "cache_full_total_mean_ms": raw["cache_full_total_mean_ms"],
            "stage4a_full_total_mean_ms": raw["optimized_full_total_mean_ms"],
            "compiled_full_total_mean_ms": raw["backend_compiled_full_total_mean_ms"],
            "cuda_graph_full_total_mean_ms": raw["backend_cuda_graph_full_total_mean_ms"],
            "backend_cuda_graph_available": backend_cuda_graph_available,
        }
    )

    row: dict[str, Any] = {
        "stage": "stage5",
        "stage5_design": "final_curated_comparison",
        "stage5_final_path_keys": _final_path_keys(
            raw["resolved_device"], backend_cuda_graph_available
        ),
        "stage5_excluded_path_keys": "backend_eager",

        # --------------------------------------------------------
        # Basic config
        # --------------------------------------------------------
        "device_requested": raw["device_requested"],
        "resolved_device": raw["resolved_device"],
        "dtype_name": raw["dtype_name"],
        "resolved_dtype": raw["resolved_dtype"],
        "batch": raw["batch"],
        "heads": raw["heads"],
        "head_dim": raw["head_dim"],
        "model_dim": raw["model_dim"],
        "prompt_len": raw["prompt_len"],
        "gen_steps": raw["gen_steps"],
        "total_seq_len": raw["total_seq_len"],
        "warmup": raw["warmup"],
        "iters": raw["iters"],
        "seed": raw["seed"],

        # --------------------------------------------------------
        # Backend status
        # --------------------------------------------------------
        "backend_compiled_status": raw["backend_compiled_status"],
        "backend_compile_mode": raw["backend_compile_mode"],
        "backend_compile_fullgraph": raw["backend_compile_fullgraph"],
        "backend_cuda_graph_status": raw["backend_cuda_graph_status"],
        "backend_cuda_graph_available": backend_cuda_graph_available,

        # --------------------------------------------------------
        # Correctness (final included paths only)
        # --------------------------------------------------------
        "naive_vs_cache_allclose": raw["naive_vs_cache_allclose"],
        "naive_vs_cache_max_abs_diff": raw["naive_vs_cache_max_abs_diff"],
        "naive_vs_cache_mean_abs_diff": raw["naive_vs_cache_mean_abs_diff"],

        "naive_vs_stage4a_allclose": raw["naive_vs_optimized_allclose"],
        "naive_vs_stage4a_max_abs_diff": raw["naive_vs_optimized_max_abs_diff"],
        "naive_vs_stage4a_mean_abs_diff": raw["naive_vs_optimized_mean_abs_diff"],

        "cache_vs_stage4a_allclose": raw["cache_vs_optimized_allclose"],
        "cache_vs_stage4a_max_abs_diff": raw["cache_vs_optimized_max_abs_diff"],
        "cache_vs_stage4a_mean_abs_diff": raw["cache_vs_optimized_mean_abs_diff"],

        "naive_vs_compiled_allclose": raw["naive_vs_backend_compiled_allclose"],
        "naive_vs_compiled_max_abs_diff": raw["naive_vs_backend_compiled_max_abs_diff"],
        "naive_vs_compiled_mean_abs_diff": raw["naive_vs_backend_compiled_mean_abs_diff"],

        "naive_vs_cuda_graph_allclose": raw["naive_vs_backend_cuda_graph_allclose"],
        "naive_vs_cuda_graph_max_abs_diff": raw["naive_vs_backend_cuda_graph_max_abs_diff"],
        "naive_vs_cuda_graph_mean_abs_diff": raw["naive_vs_backend_cuda_graph_mean_abs_diff"],

        "all_correct_final_paths": all_correct_final_paths,

        # --------------------------------------------------------
        # Memory info
        # --------------------------------------------------------
        "hidden_states_mib": raw["hidden_states_mib"],
        "separate_weights_mib": raw["separate_weights_mib"],
        "fused_weights_mib": raw["fused_weights_mib"],
        "standard_kv_cache_capacity_mib": raw["standard_kv_cache_capacity_mib"],
        "backend_kv_cache_capacity_mib": raw["backend_kv_cache_capacity_mib"],
        "backend_output_buffer_mib": raw["backend_output_buffer_mib"],

        # --------------------------------------------------------
        # Final-path total latency stats
        # --------------------------------------------------------
        "naive_full_total_mean_ms": raw["naive_full_total_mean_ms"],
        "naive_full_total_std_ms": raw["naive_full_total_std_ms"],

        "cache_full_total_mean_ms": raw["cache_full_total_mean_ms"],
        "cache_full_total_std_ms": raw["cache_full_total_std_ms"],

        "stage4a_full_total_mean_ms": raw["optimized_full_total_mean_ms"],
        "stage4a_full_total_std_ms": raw["optimized_full_total_std_ms"],

        "compiled_full_total_mean_ms": raw["backend_compiled_full_total_mean_ms"],
        "compiled_full_total_std_ms": raw["backend_compiled_full_total_std_ms"],

        "cuda_graph_full_total_mean_ms": raw["backend_cuda_graph_full_total_mean_ms"],
        "cuda_graph_full_total_std_ms": raw["backend_cuda_graph_full_total_std_ms"],

        # --------------------------------------------------------
        # Final-path amortized per-token latency
        # --------------------------------------------------------
        "naive_amortized_per_step_mean_ms": raw["naive_amortized_per_step_mean_ms"],
        "cache_amortized_per_step_mean_ms": raw["cache_amortized_per_step_mean_ms"],
        "stage4a_amortized_per_step_mean_ms": raw["optimized_amortized_per_step_mean_ms"],
        "compiled_amortized_per_step_mean_ms": raw["backend_compiled_amortized_per_step_mean_ms"],
        "cuda_graph_amortized_per_step_mean_ms": raw["backend_cuda_graph_amortized_per_step_mean_ms"],

        # --------------------------------------------------------
        # Final speedups
        # --------------------------------------------------------
        "cache_vs_naive_full_speedup": raw["cache_vs_naive_full_speedup"],

        "stage4a_vs_naive_full_speedup": raw["optimized_vs_naive_full_speedup"],
        "stage4a_vs_cache_full_speedup": raw["optimized_vs_cache_full_speedup"],

        "compiled_vs_naive_full_speedup": safe_speedup(
            raw["naive_full_total_mean_ms"],
            raw["backend_compiled_full_total_mean_ms"],
        ),
        "compiled_vs_cache_full_speedup": raw["backend_compiled_vs_cache_full_speedup"],

        "cuda_graph_vs_naive_full_speedup": safe_speedup(
            raw["naive_full_total_mean_ms"],
            raw["backend_cuda_graph_full_total_mean_ms"],
        ),
        "cuda_graph_vs_cache_full_speedup": raw["backend_cuda_graph_vs_cache_full_speedup"],
        "cuda_graph_vs_compiled_full_speedup": raw["backend_cuda_graph_vs_backend_compiled_full_speedup"],

        # --------------------------------------------------------
        # Best final path
        # --------------------------------------------------------
        "best_final_path_name": best_final_path_name,
        "best_final_path_mean_ms": best_final_path_mean_ms,
    }

    return row
