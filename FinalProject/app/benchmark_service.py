import math
from typing import Any

from src.benchmark.stage5_experiment import build_stage5_row
from app.schemas import Stage5BenchmarkRequest


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def pick(row: dict, key: str) -> Any:
    return clean_json_value(row.get(key))


def run_stage5_benchmark(req: Stage5BenchmarkRequest) -> dict:
    row = build_stage5_row(
        device_requested=req.device,
        dtype_name=req.dtype,
        batch=req.batch,
        heads=req.heads,
        head_dim=req.head_dim,
        prompt_len=req.prompt_len,
        gen_steps=req.gen_steps,
        warmup=req.warmup,
        iters=req.iters,
        seed=req.seed,
        compile_mode="reduce-overhead",
        fullgraph=False,
        enable_compile=req.enable_compile,
        enable_cuda_graphs=req.enable_cuda_graphs,
    )

    return {
        "config": {
            "device_requested": pick(row, "device_requested"),
            "resolved_device": pick(row, "resolved_device"),
            "dtype": pick(row, "dtype_name"),
            "prompt_len": pick(row, "prompt_len"),
            "gen_steps": pick(row, "gen_steps"),
            "heads": pick(row, "heads"),
            "head_dim": pick(row, "head_dim"),
        },
        "correctness": {
            "all_correct_final_paths": pick(row, "all_correct_final_paths"),
            "naive_vs_cache_allclose": pick(row, "naive_vs_cache_allclose"),
            "naive_vs_stage4a_allclose": pick(row, "naive_vs_stage4a_allclose"),
        },
        "latency_ms": {
            "naive": pick(row, "naive_full_total_mean_ms"),
            "cache": pick(row, "cache_full_total_mean_ms"),
            "stage4a": pick(row, "stage4a_full_total_mean_ms"),
            "compiled": pick(row, "compiled_full_total_mean_ms"),
            "cuda_graph": pick(row, "cuda_graph_full_total_mean_ms"),
        },
        "speedups": {
            "cache_vs_naive": pick(row, "cache_vs_naive_full_speedup"),
            "stage4a_vs_cache": pick(row, "stage4a_vs_cache_full_speedup"),
            "compiled_vs_cache": pick(row, "compiled_vs_cache_full_speedup"),
            "cuda_graph_vs_cache": pick(row, "cuda_graph_vs_cache_full_speedup"),
        },
        "best_path": {
            "name": pick(row, "best_final_path_name"),
            "latency_ms": pick(row, "best_final_path_mean_ms"),
        },
    }