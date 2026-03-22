# src/benchmark/stage4b_experiment.py

import csv
import math
import statistics
import time
from typing import Any, Callable, Optional

import torch

from src.attention.backend_decode import (
    make_compiled_backend_runner,
    make_preallocated_backend_runner,
)
from src.attention.cuda_graph_decode import make_cuda_graph_backend_runner
from src.attention.decode import (
    create_projection_weights,
    run_kv_cache_decode,
    run_naive_decode,
)
from src.attention.optimized_decode import (
    build_fused_projection_weights_from_separate,
    run_optimized_kv_cache_decode,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_requested: str) -> torch.device:
    if device_requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA, but CUDA is not available.")

    return torch.device(device_requested)


def resolve_dtype(dtype_name: str) -> torch.dtype:
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark a callable and return summary stats in milliseconds.

    Note:
    - warmup is intentionally excluded from timing
    - this is especially important for torch.compile, where warmup can include compile cost
    """
    if iters <= 0:
        raise ValueError("iters must be > 0")

    times_ms = []
    last_output = None

    with torch.no_grad():
        for _ in range(warmup):
            last_output = fn()
        sync_if_needed(device)

        for _ in range(iters):
            sync_if_needed(device)
            t0 = time.perf_counter()
            last_output = fn()
            sync_if_needed(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "checksum": checksum(last_output),
    }


def nan_stats() -> dict[str, float]:
    return {
        "mean_ms": float("nan"),
        "std_ms": float("nan"),
        "min_ms": float("nan"),
        "max_ms": float("nan"),
        "checksum": float("nan"),
    }


def flatten_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
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


def checksum(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return float("nan")
    if x.numel() == 0:
        return 0.0
    return float(x.float().sum().item())


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-4
    if dtype == torch.bfloat16:
        return 1e-2, 5e-2
    if dtype == torch.float16:
        return 5e-3, 5e-2
    return 1e-5, 1e-4


def compare_tensors(
    a: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """
    Compare two tensors.
    If one path is unavailable, return NaNs.
    """
    if a is None or b is None:
        return {
            "available": False,
            "allclose": True,  # unavailable path should not fail "available paths" correctness
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
        }

    if a.shape != b.shape:
        return {
            "available": True,
            "allclose": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
        }

    if a.numel() == 0:
        return {
            "available": True,
            "allclose": True,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }

    diff = (a - b).abs()

    return {
        "available": True,
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def safe_speedup(baseline_ms: float, improved_ms: float) -> float:
    """
    baseline / improved
    """
    if math.isnan(baseline_ms) or math.isnan(improved_ms):
        return float("nan")
    if improved_ms <= 0.0:
        return float("nan")
    return baseline_ms / improved_ms


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

    This compares:
    - naive decode
    - Stage 2/3 KV-cache decode
    - Stage 4A optimized decode
    - Stage 4B-A eager backend path
    - Stage 4B-B compiled backend path
    - Stage 4B-C CUDA Graph backend path (if available)
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

    # ------------------------------------------------------------
    # Build Stage 4B runners once
    # ------------------------------------------------------------
    backend_eager_runner = make_preallocated_backend_runner(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        device=device,
        dtype=dtype,
    )

    backend_compiled_runner, backend_compiled_status = make_compiled_backend_runner(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        fused_weights=fused_weights,
        device=device,
        dtype=dtype,
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

    # ------------------------------------------------------------
    # Correctness checks
    # ------------------------------------------------------------
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
            # clone only for correctness bookkeeping so later replays do not overwrite
            backend_cuda_graph_out = backend_cuda_graph_runner(hidden_states).clone()

    naive_vs_cache = compare_tensors(naive_out, cache_out, atol=atol, rtol=rtol)
    naive_vs_optimized = compare_tensors(naive_out, optimized_out, atol=atol, rtol=rtol)
    cache_vs_optimized = compare_tensors(cache_out, optimized_out, atol=atol, rtol=rtol)

    naive_vs_backend_eager = compare_tensors(
        naive_out, backend_eager_out, atol=atol, rtol=rtol
    )
    naive_vs_backend_compiled = compare_tensors(
        naive_out, backend_compiled_out, atol=atol, rtol=rtol
    )
    naive_vs_backend_cuda_graph = compare_tensors(
        naive_out, backend_cuda_graph_out, atol=atol, rtol=rtol
    )

    backend_eager_vs_optimized = compare_tensors(
        backend_eager_out, optimized_out, atol=atol, rtol=rtol
    )
    backend_compiled_vs_backend_eager = compare_tensors(
        backend_compiled_out, backend_eager_out, atol=atol, rtol=rtol
    )
    backend_cuda_graph_vs_backend_compiled = compare_tensors(
        backend_cuda_graph_out, backend_compiled_out, atol=atol, rtol=rtol
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

    # ------------------------------------------------------------
    # Memory bookkeeping
    # ------------------------------------------------------------
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

    backend_kv_cache_capacity_bytes = standard_kv_cache_capacity_bytes

    backend_output_buffer_bytes = batch * heads * gen_steps * head_dim * element_size

    # ------------------------------------------------------------
    # Benchmark all paths
    # ------------------------------------------------------------
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

    backend_eager_total_stats = benchmark_callable(
        fn=lambda: backend_eager_runner(hidden_states),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    backend_compiled_total_stats = benchmark_callable(
        fn=lambda: backend_compiled_runner(hidden_states),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    if backend_cuda_graph_runner is not None:
        backend_cuda_graph_total_stats = benchmark_callable(
            fn=lambda: backend_cuda_graph_runner(hidden_states),
            warmup=warmup,
            iters=iters,
            device=device,
        )
    else:
        backend_cuda_graph_total_stats = nan_stats()

    # ------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------
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

    row: dict[str, Any] = {
        "stage": "stage4b",
        "stage4a_path_name": "fused_qkv_manual_single_query",
        "stage4b_eager_path_name": "tensor_only_preallocated_eager",
        "stage4b_compiled_path_name": "tensor_only_preallocated_compile",
        "stage4b_cuda_graph_path_name": "tensor_only_preallocated_cuda_graph",

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
        "naive_vs_backend_compiled_max_abs_diff": naive_vs_backend_compiled["max_abs_diff"],
        "naive_vs_backend_compiled_mean_abs_diff": naive_vs_backend_compiled["mean_abs_diff"],

        "naive_vs_backend_cuda_graph_allclose": naive_vs_backend_cuda_graph["allclose"],
        "naive_vs_backend_cuda_graph_max_abs_diff": naive_vs_backend_cuda_graph["max_abs_diff"],
        "naive_vs_backend_cuda_graph_mean_abs_diff": naive_vs_backend_cuda_graph["mean_abs_diff"],

        "backend_eager_vs_optimized_allclose": backend_eager_vs_optimized["allclose"],
        "backend_eager_vs_optimized_max_abs_diff": backend_eager_vs_optimized["max_abs_diff"],
        "backend_eager_vs_optimized_mean_abs_diff": backend_eager_vs_optimized["mean_abs_diff"],

        "backend_compiled_vs_backend_eager_allclose": backend_compiled_vs_backend_eager["allclose"],
        "backend_compiled_vs_backend_eager_max_abs_diff": backend_compiled_vs_backend_eager["max_abs_diff"],
        "backend_compiled_vs_backend_eager_mean_abs_diff": backend_compiled_vs_backend_eager["mean_abs_diff"],

        "backend_cuda_graph_vs_backend_compiled_allclose": backend_cuda_graph_vs_backend_compiled["allclose"],
        "backend_cuda_graph_vs_backend_compiled_max_abs_diff": backend_cuda_graph_vs_backend_compiled["max_abs_diff"],
        "backend_cuda_graph_vs_backend_compiled_mean_abs_diff": backend_cuda_graph_vs_backend_compiled["mean_abs_diff"],

        "all_correct_available_paths": all_correct_available_paths,

        "hidden_states_bytes": hidden_states_bytes,
        "hidden_states_mib": bytes_to_mib(hidden_states_bytes),

        "separate_weights_bytes": separate_weights_bytes,
        "separate_weights_mib": bytes_to_mib(separate_weights_bytes),

        "fused_weights_bytes": fused_weights_bytes,
        "fused_weights_mib": bytes_to_mib(fused_weights_bytes),

        "standard_kv_cache_capacity_bytes": standard_kv_cache_capacity_bytes,
        "standard_kv_cache_capacity_mib": bytes_to_mib(standard_kv_cache_capacity_bytes),

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
        "backend_eager_amortized_per_step_mean_ms": backend_eager_amortized_per_step_mean_ms,
        "backend_compiled_amortized_per_step_mean_ms": backend_compiled_amortized_per_step_mean_ms,
        "backend_cuda_graph_amortized_per_step_mean_ms": backend_cuda_graph_amortized_per_step_mean_ms,

        "cache_vs_naive_full_speedup": safe_speedup(
            naive_total_stats["mean_ms"], cache_total_stats["mean_ms"]
        ),
        "optimized_vs_naive_full_speedup": safe_speedup(
            naive_total_stats["mean_ms"], optimized_total_stats["mean_ms"]
        ),
        "optimized_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"], optimized_total_stats["mean_ms"]
        ),

        "backend_eager_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"], backend_eager_total_stats["mean_ms"]
        ),
        "backend_compiled_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"], backend_compiled_total_stats["mean_ms"]
        ),
        "backend_cuda_graph_vs_cache_full_speedup": safe_speedup(
            cache_total_stats["mean_ms"], backend_cuda_graph_total_stats["mean_ms"]
        ),

        "backend_compiled_vs_backend_eager_full_speedup": safe_speedup(
            backend_eager_total_stats["mean_ms"], backend_compiled_total_stats["mean_ms"]
        ),
        "backend_cuda_graph_vs_backend_compiled_full_speedup": safe_speedup(
            backend_compiled_total_stats["mean_ms"], backend_cuda_graph_total_stats["mean_ms"]
        ),
    }

    return row


def write_rows_to_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    """
    Write flat experiment rows to CSV.
    """
    if len(rows) == 0:
        raise ValueError("rows is empty; nothing to write.")

    fieldnames = sorted(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)