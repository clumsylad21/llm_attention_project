from __future__ import annotations

from src.benchmark.config import BenchmarkConfig
from src.benchmark.result import BenchmarkResult
from src.benchmark.stage5_experiment import build_stage5_row
from src.benchmark.stage6_experiment import build_stage6_row


def run_stage5_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run Stage 5 through the existing benchmark implementation.

    This wrapper is intentionally thin: it owns the stable internal API surface,
    while Stage 5 still owns benchmark setup, timing, and correctness checks.
    """
    row = build_stage5_row(**config.to_stage5_kwargs())
    return BenchmarkResult.from_stage5_row(row)


def run_stage6_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run Stage 6 through the existing benchmark implementation.

    Stage 6 already reuses Stage 5 internally, so this service only adapts the
    row dict into the structured result shape expected by future API/dashboard code.
    """
    row = build_stage6_row(**config.to_stage6_kwargs())
    return BenchmarkResult.from_stage6_row(row)
