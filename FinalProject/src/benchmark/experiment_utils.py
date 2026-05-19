# src/benchmark/experiment_utils.py

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable, Optional

import torch

from src.benchmark.csv_utils import write_rows_to_csv as _write_rows_to_csv


def set_seed(seed: int) -> None:
    """
    Make synthetic experiments reproducible.
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
    CUDA kernels are asynchronous, so synchronize before/after timing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checksum(x: Any) -> float:
    """
    Small scalar sanity summary for tensor benchmark outputs.

    Some benchmarked functions return helper/cache objects instead of tensors.
    For those cases, checksum is not meaningful, so return NaN.
    """
    if x is None:
        return float("nan")

    if not isinstance(x, torch.Tensor):
        return float("nan")

    if x.numel() == 0:
        return 0.0

    return float(x.float().sum().item())


def benchmark_callable(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark a callable and return summary stats in milliseconds.

    Warmup iterations are excluded from timing. This is especially useful
    for torch.compile paths, where warmup may include compile overhead.
    """
    if iters <= 0:
        raise ValueError("iters must be > 0")

    times_ms: list[float] = []
    last_output: Any = None

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

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "checksum": checksum(last_output),
    }


def nan_stats() -> dict[str, float]:
    """
    Standard placeholder stats for unavailable optional paths.
    """
    return {
        "mean_ms": float("nan"),
        "std_ms": float("nan"),
        "min_ms": float("nan"),
        "max_ms": float("nan"),
        "checksum": float("nan"),
    }


def flatten_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    """
    Convert a benchmark stats dict into flat CSV-friendly fields.
    """
    return {
        f"{prefix}_mean_ms": stats["mean_ms"],
        f"{prefix}_std_ms": stats["std_ms"],
        f"{prefix}_min_ms": stats["min_ms"],
        f"{prefix}_max_ms": stats["max_ms"],
    }


def tensor_bytes(x: torch.Tensor) -> int:
    """
    Return the number of bytes used by a tensor.
    """
    return x.numel() * x.element_size()


def bytes_to_mib(num_bytes: int) -> float:
    """
    Convert bytes to MiB.
    """
    return num_bytes / (1024.0 * 1024.0)


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """
    Use slightly looser correctness tolerances for low-precision dtypes.
    """
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
    Compare two tensors and return correctness/debug metrics.

    If one path is unavailable, return NaNs but mark allclose=True so that
    optional unavailable paths do not fail the overall correctness flag.
    """
    if a is None or b is None:
        return {
            "available": False,
            "allclose": True,
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
    Return baseline / improved, or NaN if the ratio is not meaningful.
    """
    if math.isnan(baseline_ms) or math.isnan(improved_ms):
        return float("nan")

    if improved_ms <= 0.0:
        return float("nan")

    return baseline_ms / improved_ms


def write_rows_to_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    """
    Backward-compatible wrapper around csv_utils.write_rows_to_csv.

    Existing stage scripts use:
        write_rows_to_csv(rows, csv_path)

    csv_utils uses:
        write_rows_to_csv(csv_path, rows)
    """
    _write_rows_to_csv(csv_path, rows)
