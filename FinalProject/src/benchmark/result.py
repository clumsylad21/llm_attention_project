from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, dict):
        return {key: _clean_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]

    return value


@dataclass(frozen=True)
class BackendResult:
    name: str
    mean_ms: float
    std_ms: float
    available: bool
    allclose: Optional[bool] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    status: Optional[str] = None
    correctness_reference: Optional[str] = None

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        name: str,
        latency_prefix: str,
        correctness_prefix: Optional[str] = None,
        available: Optional[bool] = None,
        status_key: Optional[str] = None,
        correctness_reference: Optional[str] = "naive",
    ) -> "BackendResult":
        mean_ms = row.get(f"{latency_prefix}_mean_ms", float("nan"))
        std_ms = row.get(f"{latency_prefix}_std_ms", float("nan"))

        if available is None:
            available = _is_finite_number(mean_ms)

        if correctness_prefix is None:
            allclose = True if available else None
            max_abs_diff = 0.0 if available else None
            mean_abs_diff = 0.0 if available else None
            correctness_reference = None
        else:
            allclose = row.get(f"{correctness_prefix}_allclose")
            max_abs_diff = row.get(f"{correctness_prefix}_max_abs_diff")
            mean_abs_diff = row.get(f"{correctness_prefix}_mean_abs_diff")

        status = row.get(status_key) if status_key is not None else None

        return cls(
            name=name,
            mean_ms=mean_ms,
            std_ms=std_ms,
            available=bool(available),
            allclose=allclose,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            status=status,
            correctness_reference=correctness_reference,
        )

    def to_dict(self, *, clean_nan: bool = True) -> dict[str, Any]:
        data = asdict(self)
        return _clean_json_value(data) if clean_nan else data


@dataclass(frozen=True)
class BenchmarkResult:
    stage: str
    device: str
    dtype: str
    best_backend: str
    best_mean_ms: float
    backends: list[BackendResult]
    raw: dict[str, Any]
    all_correct: Optional[bool] = None

    @classmethod
    def from_stage5_row(cls, row: dict[str, Any]) -> "BenchmarkResult":
        backends = _stage5_backends(row)

        return cls(
            stage="stage5",
            device=row.get("resolved_device", row.get("device_requested", "unknown")),
            dtype=row.get("resolved_dtype", row.get("dtype_name", "unknown")),
            best_backend=row.get("best_final_path_name", "unknown"),
            best_mean_ms=row.get("best_final_path_mean_ms", float("nan")),
            backends=backends,
            raw=row,
            all_correct=row.get("all_correct_final_paths"),
        )

    @classmethod
    def from_stage6_row(cls, row: dict[str, Any]) -> "BenchmarkResult":
        backends = _stage5_backends(row)
        backends.append(
            BackendResult.from_row(
                row,
                name="stage6_custom_cuda",
                latency_prefix="stage6_full_total",
                correctness_prefix="stage6_vs_naive",
                available=bool(row.get("stage6_available", False)),
                status_key="stage6_status",
                correctness_reference="naive",
            )
        )

        return cls(
            stage="stage6",
            device=row.get("resolved_device", row.get("device_requested", "unknown")),
            dtype=row.get("resolved_dtype", row.get("dtype_name", "unknown")),
            best_backend=row.get("best_path_with_stage6_name", "unknown"),
            best_mean_ms=row.get("best_path_with_stage6_mean_ms", float("nan")),
            backends=backends,
            raw=row,
            all_correct=row.get("all_correct_with_stage6"),
        )

    def to_dict(
        self,
        *,
        clean_nan: bool = True,
        include_raw: bool = True,
    ) -> dict[str, Any]:
        data = {
            "stage": self.stage,
            "device": self.device,
            "dtype": self.dtype,
            "best_backend": self.best_backend,
            "best_mean_ms": self.best_mean_ms,
            "all_correct": self.all_correct,
            "backends": [
                backend.to_dict(clean_nan=False) for backend in self.backends
            ],
        }

        if include_raw:
            data["raw"] = self.raw

        return _clean_json_value(data) if clean_nan else data


def _stage5_backends(row: dict[str, Any]) -> list[BackendResult]:
    return [
        BackendResult.from_row(
            row,
            name="naive",
            latency_prefix="naive_full_total",
        ),
        BackendResult.from_row(
            row,
            name="cache",
            latency_prefix="cache_full_total",
            correctness_prefix="naive_vs_cache",
        ),
        BackendResult.from_row(
            row,
            name="stage4a",
            latency_prefix="stage4a_full_total",
            correctness_prefix="naive_vs_stage4a",
        ),
        BackendResult.from_row(
            row,
            name="compiled",
            latency_prefix="compiled_full_total",
            correctness_prefix="naive_vs_compiled",
            status_key="backend_compiled_status",
        ),
        BackendResult.from_row(
            row,
            name="cuda_graph",
            latency_prefix="cuda_graph_full_total",
            correctness_prefix="naive_vs_cuda_graph",
            available=bool(row.get("backend_cuda_graph_available", False)),
            status_key="backend_cuda_graph_status",
        ),
    ]
