from __future__ import annotations

import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.benchmark.config import BenchmarkConfig
from src.benchmark.result import BenchmarkResult


_DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "results" / "runs"


def make_run_id(stage: str, now: datetime | None = None) -> str:
    """
    Build a filesystem-friendly run id with enough timestamp precision for tests
    and local benchmark usage.
    """
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage.strip()).strip("-")
    timestamp = current_time.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{safe_stage}"


def collect_environment() -> dict[str, Any]:
    if torch is None:
        return {
            "current_time": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "cuda_device_names": [],
            "cwd": str(Path.cwd()),
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    device_names: list[str] = []
    if cuda_available:
        device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]

    return {
        "current_time": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": device_count,
        "cuda_device_names": device_names,
        "cwd": str(Path.cwd()),
    }


def save_benchmark_run(
    config: BenchmarkConfig,
    result: BenchmarkResult,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    runs_dir = _resolve_base_dir(base_dir)
    run_id = make_run_id(result.stage)
    run_dir = runs_dir / run_id

    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{run_id}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)

    files = {
        "config": run_dir / "config.json",
        "metrics": run_dir / "metrics.json",
        "raw_row": run_dir / "raw_row.json",
        "environment": run_dir / "environment.json",
    }

    _write_json(files["config"], config.to_dict())
    _write_json(files["metrics"], result.to_dict(include_raw=False))
    _write_json(files["raw_row"], result.raw)
    _write_json(files["environment"], collect_environment())

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "files": {name: str(path) for name, path in files.items()},
    }


def get_latest_run_dir(base_dir: Path | str | None = None) -> Path | None:
    runs_dir = _resolve_base_dir(base_dir)
    if not runs_dir.exists():
        return None

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda path: path.stat().st_mtime_ns)


def load_latest_metrics(base_dir: Path | str | None = None) -> dict[str, Any] | None:
    latest_run_dir = get_latest_run_dir(base_dir)
    if latest_run_dir is None:
        return None

    metrics_path = latest_run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_runs(
    limit: int = 20,
    base_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    runs_dir = _resolve_base_dir(base_dir)
    if not runs_dir.exists():
        return []

    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs[:limit]:
        metrics = _load_json_file(run_dir / "metrics.json") or {}
        environment = _load_json_file(run_dir / "environment.json") or {}

        runs.append(
            {
                "run_id": run_dir.name,
                "stage": metrics.get("stage"),
                "device": metrics.get("device"),
                "dtype": metrics.get("dtype"),
                "best_backend": metrics.get("best_backend"),
                "best_mean_ms": metrics.get("best_mean_ms"),
                "all_correct": metrics.get("all_correct"),
                "created_at": environment.get("current_time"),
            }
        )

    return runs


def get_run_dir(
    run_id: str,
    base_dir: Path | str | None = None,
) -> Path | None:
    if not _is_safe_run_id(run_id):
        return None

    run_dir = _resolve_base_dir(base_dir) / run_id
    if not run_dir.is_dir():
        return None

    return run_dir


def load_run_metrics(
    run_id: str,
    base_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    run_dir = get_run_dir(run_id, base_dir)
    if run_dir is None:
        return None

    return _load_json_file(run_dir / "metrics.json")


def load_run_config(
    run_id: str,
    base_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    run_dir = get_run_dir(run_id, base_dir)
    if run_dir is None:
        return None

    return _load_json_file(run_dir / "config.json")


def load_run_raw_row(
    run_id: str,
    base_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    run_dir = get_run_dir(run_id, base_dir)
    if run_dir is None:
        return None

    return _load_json_file(run_dir / "raw_row.json")


def load_run_environment(
    run_id: str,
    base_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    run_dir = get_run_dir(run_id, base_dir)
    if run_dir is None:
        return None

    return _load_json_file(run_dir / "environment.json")


def _resolve_base_dir(base_dir: Path | str | None) -> Path:
    return Path(base_dir) if base_dir is not None else _DEFAULT_RUNS_DIR


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(data),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_safe_run_id(run_id: str) -> bool:
    if not isinstance(run_id, str) or not run_id:
        return False

    return Path(run_id).name == run_id and run_id not in {".", ".."}

def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {_json_safe_key(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def _json_safe_key(key: Any) -> str:
    if isinstance(key, str):
        return key

    if isinstance(key, (int, float, bool)) or key is None:
        return str(key)

    return repr(key)
