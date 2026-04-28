import csv
import statistics
import time
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


def set_seed(seed: int) -> None:
    """
    Keep experiments reproducible.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_requested: str) -> torch.device:
    """
    Resolve auto/cpu/cuda into an actual torch.device.
    """
    if device_requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA, but CUDA is not available.")

    return torch.device(device_requested)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """
    Map short dtype names to torch dtypes.
    """
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Choose from: {list(mapping.keys())}"
        )

    return mapping[dtype_name]


def sync_if_needed(device: torch.device) -> None:
    """
    CUDA timings need explicit synchronization.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    fn,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark one callable and return summary stats in milliseconds.
    """
    if iters <= 0:
        raise ValueError("iters must be > 0")

    times_ms = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
            sync_if_needed(device)

        for _ in range(iters):
            sync_if_needed(device)
            t0 = time.perf_counter()

            _ = fn()

            sync_if_needed(device)
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }


def flatten_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    """
    Convert stats dict into flat CSV-friendly keys.
    """
    return {
        f"{prefix}_mean_ms": stats["mean_ms"],
        f"{prefix}_std_ms": stats["std_ms"],
        f"{prefix}_min_ms": stats["min_ms"],
        f"{prefix}_max_ms": stats["max_ms"],
    }


def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def checksum(x: torch.Tensor) -> float:
    """
    Small scalar sanity summary.
    """
    if x.numel() == 0:
        return 0.0
    return float(x.float().sum().item())


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """
    Use slightly looser tolerances for low-precision dtypes.
    """
    if dtype == torch.float32:
        return 1e-5, 1e-4
    if dtype == torch.bfloat16:
        return 1e-2, 5e-2
    if dtype == torch.float16:
        return 5e-3, 5e-2
    return 1e-5, 1e-4


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """
    Compare two outputs numerically.
    """
    if a.shape != b.shape:
        return {
            "allclose": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
        }

    if a.numel() == 0:
        return {
            "allclose": True,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }

    diff = (a - b).abs()

    return {
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def safe_speedup(baseline_ms: float, improved_ms: float) -> float:
    """
    baseline / improved
    """
    if improved_ms <= 0.0:
        return 0.0
    return baseline_ms / improved_ms


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
        "optimized_kv_cache_capacity_mib": bytes_to_mib(optimized_kv_cache_capacity_bytes),

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


def write_rows_to_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    """
    Write flat experiment rows to CSV.
    """
    if len(rows) == 0:
        raise ValueError("No rows to write.")

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)