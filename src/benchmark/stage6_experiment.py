# src/benchmark/stage6_experiment.py

from __future__ import annotations

import math
from typing import Any

import torch

from src.attention.decode import create_projection_weights, run_naive_decode, run_kv_cache_decode
from src.attention.optimized_decode import (
    build_fused_projection_weights_from_separate,
    run_optimized_kv_cache_decode,
)
from src.attention.stage6_cuda_extension import STAGE6_TILE_TOKENS, stage6_availability
from src.attention.stage6_custom_decode import run_stage6_custom_decode
from src.benchmark.stage4b_experiment import (
    benchmark_callable,
    compare_tensors,
    default_tolerances,
    nan_stats,
    resolve_device,
    resolve_dtype,
    safe_speedup,
    set_seed,
    write_rows_to_csv,
)
from src.benchmark.stage5_experiment import build_stage5_row


def _best_path_with_stage6(row: dict[str, Any]) -> tuple[str, float]:
    candidates = [
        ("naive", row["naive_full_total_mean_ms"]),
        ("cache", row["cache_full_total_mean_ms"]),
        ("stage4a", row["stage4a_full_total_mean_ms"]),
        ("compiled", row["compiled_full_total_mean_ms"]),
    ]

    if row["backend_cuda_graph_available"]:
        candidates.append(("cuda_graph", row["cuda_graph_full_total_mean_ms"]))

    if row["stage6_available"]:
        candidates.append(("stage6_custom_cuda", row["stage6_full_total_mean_ms"]))

    valid_candidates = []
    for name, value in candidates:
        if not math.isnan(value):
            valid_candidates.append((name, value))

    if len(valid_candidates) == 0:
        return "unknown", float("nan")

    return min(valid_candidates, key=lambda x: x[1])


def build_stage6_row(
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
    enable_stage6: bool = True,
) -> dict[str, Any]:
    """
    Build one Stage 6 experiment row.

    Design choice:
    - Reuse the Stage 5 final-comparison layer as the baseline story.
    - Add one focused custom CUDA path on top.
    - Keep Stage 6 custom work limited to the single-query attention core.
    """
    stage5_row = build_stage5_row(
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

    device = resolve_device(device_requested)
    dtype = resolve_dtype(dtype_name)
    model_dim = heads * head_dim
    total_seq_len = prompt_len + gen_steps

    # Recreate the exact synthetic input/weights used by earlier stages.
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

    stage6_available, stage6_status = stage6_availability(
        device=device,
        dtype=dtype,
        enable_stage6=enable_stage6,
    )

    with torch.no_grad():
        naive_out = run_naive_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            separate_weights,
            heads,
            head_dim,
        )

        stage4a_out = run_optimized_kv_cache_decode(
            hidden_states,
            prompt_len,
            gen_steps,
            fused_weights,
        )

        stage6_out = None
        if stage6_available:
            stage6_out = run_stage6_custom_decode(
                hidden_states,
                prompt_len,
                gen_steps,
                fused_weights,
            )

    stage6_vs_naive = compare_tensors(
        stage6_out,
        naive_out,
        atol=atol,
        rtol=rtol,
    )
    stage6_vs_stage4a = compare_tensors(
        stage6_out,
        stage4a_out,
        atol=atol,
        rtol=rtol,
    )

    if stage6_available:
        stage6_total_stats = benchmark_callable(
            fn=lambda: run_stage6_custom_decode(
                hidden_states,
                prompt_len,
                gen_steps,
                fused_weights,
            ),
            warmup=warmup,
            iters=iters,
            device=device,
        )
    else:
        stage6_total_stats = nan_stats()

    stage6_amortized_per_step_mean_ms = (
        stage6_total_stats["mean_ms"] / gen_steps if gen_steps > 0 else 0.0
    )

    all_correct_with_stage6 = (
        bool(stage5_row["all_correct_final_paths"])
        and bool(stage6_vs_naive["allclose"])
        and bool(stage6_vs_stage4a["allclose"])
    )

    best_path_with_stage6_name, best_path_with_stage6_mean_ms = _best_path_with_stage6(
        {
            **stage5_row,
            "stage6_available": stage6_available,
            "stage6_full_total_mean_ms": stage6_total_stats["mean_ms"],
        }
    )

    row: dict[str, Any] = dict(stage5_row)
    row.update(
        {
            "stage": "stage6",
            "stage6_design": "custom_cuda_attention_core_only",
            "stage6_scope": "single_query_forward_only",
            "stage6_custom_work_boundary": (
                "custom_cuda_only_for_attention_inner_loop;"
                "projection_cache_and_benchmark_remain_pytorch"
            ),
            "stage6_tile_tokens": STAGE6_TILE_TOKENS,
            "stage6_available": stage6_available,
            "stage6_status": stage6_status,
            "stage6_path_name": "custom_cuda_tiled_single_query_attention",
            "stage6_vs_naive_allclose": stage6_vs_naive["allclose"],
            "stage6_vs_naive_max_abs_diff": stage6_vs_naive["max_abs_diff"],
            "stage6_vs_naive_mean_abs_diff": stage6_vs_naive["mean_abs_diff"],
            "stage6_vs_stage4a_allclose": stage6_vs_stage4a["allclose"],
            "stage6_vs_stage4a_max_abs_diff": stage6_vs_stage4a["max_abs_diff"],
            "stage6_vs_stage4a_mean_abs_diff": stage6_vs_stage4a["mean_abs_diff"],
            "stage6_full_total_mean_ms": stage6_total_stats["mean_ms"],
            "stage6_full_total_std_ms": stage6_total_stats["std_ms"],
            "stage6_full_total_min_ms": stage6_total_stats["min_ms"],
            "stage6_full_total_max_ms": stage6_total_stats["max_ms"],
            "stage6_checksum": stage6_total_stats["checksum"] if stage6_available else float("nan"),
            "stage6_amortized_per_step_mean_ms": stage6_amortized_per_step_mean_ms,
            "stage6_vs_cache_full_speedup": safe_speedup(
                stage5_row["cache_full_total_mean_ms"],
                stage6_total_stats["mean_ms"],
            ),
            "stage6_vs_stage4a_full_speedup": safe_speedup(
                stage5_row["stage4a_full_total_mean_ms"],
                stage6_total_stats["mean_ms"],
            ),
            "stage6_vs_compiled_full_speedup": safe_speedup(
                stage5_row["compiled_full_total_mean_ms"],
                stage6_total_stats["mean_ms"],
            ),
            "stage6_vs_cuda_graph_full_speedup": safe_speedup(
                stage5_row["cuda_graph_full_total_mean_ms"],
                stage6_total_stats["mean_ms"],
            ),
            "stage6_vs_best_stage5_full_speedup": safe_speedup(
                stage5_row["best_final_path_mean_ms"],
                stage6_total_stats["mean_ms"],
            ),
            "all_correct_with_stage6": all_correct_with_stage6,
            "best_path_with_stage6_name": best_path_with_stage6_name,
            "best_path_with_stage6_mean_ms": best_path_with_stage6_mean_ms,
        }
    )
    return row
