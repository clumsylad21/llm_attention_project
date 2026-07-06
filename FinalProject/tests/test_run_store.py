from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.benchmark.config import BenchmarkConfig
from src.benchmark.result import BackendResult, BenchmarkResult
from src.benchmark.run_store import (
    get_latest_run_dir,
    get_run_dir,
    list_runs,
    load_latest_metrics,
    load_run_config,
    load_run_environment,
    load_run_metrics,
    load_run_raw_row,
    make_run_id,
    save_benchmark_run,
)


def _tiny_cpu_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        device_requested="cpu",
        dtype_name="fp32",
        batch=1,
        heads=2,
        head_dim=16,
        prompt_len=16,
        gen_steps=2,
        warmup=0,
        iters=1,
        enable_compile=False,
        enable_cuda_graphs=False,
        enable_stage6=False,
    )


def _sample_stage5_result() -> BenchmarkResult:
    raw = {
        "resolved_device": "cpu",
        "resolved_dtype": "fp32",
        "naive_full_total_mean_ms": 1.0,
        "naive_full_total_std_ms": 0.0,
        "cache_full_total_mean_ms": 0.8,
        "cache_full_total_std_ms": 0.0,
        "naive_vs_cache_allclose": True,
        "naive_vs_cache_max_abs_diff": 0.0,
        "naive_vs_cache_mean_abs_diff": 0.0,
        "all_correct_final_paths": True,
        "best_final_path_name": "cache",
        "best_final_path_mean_ms": 0.8,
    }

    return BenchmarkResult(
        stage="stage5",
        device="cpu",
        dtype="fp32",
        best_backend="cache",
        best_mean_ms=0.8,
        all_correct=True,
        backends=[
            BackendResult(
                name="naive",
                mean_ms=1.0,
                std_ms=0.0,
                available=True,
            ),
            BackendResult(
                name="cache",
                mean_ms=0.8,
                std_ms=0.0,
                available=True,
                allclose=True,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
            ),
        ],
        raw=raw,
    )


def test_save_benchmark_run_writes_expected_files(tmp_path: Path) -> None:
    config = _tiny_cpu_config()
    result = _sample_stage5_result()

    saved = save_benchmark_run(config, result, base_dir=tmp_path)
    run_dir = Path(saved["run_dir"])

    for filename in ("config.json", "metrics.json", "raw_row.json", "environment.json"):
        assert (run_dir / filename).exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage"] == "stage5"
    assert metrics["all_correct"] is True
    assert "raw" not in metrics

    raw_row = json.loads((run_dir / "raw_row.json").read_text(encoding="utf-8"))
    assert raw_row["best_final_path_name"] == "cache"
    assert raw_row["naive_full_total_mean_ms"] == 1.0


def test_latest_run_helpers_return_saved_metrics(tmp_path: Path) -> None:
    result = _sample_stage5_result()
    saved = save_benchmark_run(_tiny_cpu_config(), result, base_dir=tmp_path)

    assert get_latest_run_dir(base_dir=tmp_path) == Path(saved["run_dir"])

    metrics = load_latest_metrics(base_dir=tmp_path)
    assert metrics is not None
    assert metrics["stage"] == "stage5"
    assert metrics["best_backend"] == "cache"


def test_non_finite_floats_are_written_as_null(tmp_path: Path) -> None:
    result = BenchmarkResult(
        stage="stage5",
        device="cpu",
        dtype="fp32",
        best_backend="cache",
        best_mean_ms=math.inf,
        all_correct=True,
        backends=[
            BackendResult(
                name="cache",
                mean_ms=math.inf,
                std_ms=math.nan,
                available=True,
                max_abs_diff=-math.inf,
            )
        ],
        raw={
            "finite": 1.25,
            "nan_value": math.nan,
            "inf_value": math.inf,
            "nested": {"neg_inf": -math.inf},
        },
    )

    saved = save_benchmark_run(_tiny_cpu_config(), result, base_dir=tmp_path)
    run_dir = Path(saved["run_dir"])

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_mean_ms"] is None
    assert metrics["backends"][0]["mean_ms"] is None
    assert metrics["backends"][0]["std_ms"] is None
    assert metrics["backends"][0]["max_abs_diff"] is None

    raw_row = json.loads((run_dir / "raw_row.json").read_text(encoding="utf-8"))
    assert raw_row["finite"] == 1.25
    assert raw_row["nan_value"] is None
    assert raw_row["inf_value"] is None
    assert raw_row["nested"]["neg_inf"] is None


def test_make_run_id_is_filesystem_friendly() -> None:
    run_id = make_run_id("stage 5 / cpu")
    assert "/" not in run_id
    assert " " not in run_id
    assert run_id.endswith("_stage-5-cpu")


def test_save_tiny_stage5_cpu_benchmark_when_stage5_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("src.benchmark.stage5_experiment")

    from src.benchmark.service import run_stage5_benchmark

    result = run_stage5_benchmark(_tiny_cpu_config())
    saved = save_benchmark_run(_tiny_cpu_config(), result, base_dir=tmp_path)
    run_dir = Path(saved["run_dir"])

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage"] == "stage5"
    assert metrics["all_correct"] is True
    assert "raw" not in metrics

    raw_row = json.loads((run_dir / "raw_row.json").read_text(encoding="utf-8"))
    assert "naive_full_total_mean_ms" in raw_row


def test_load_run_artifacts_by_run_id(tmp_path: Path) -> None:
    result = _sample_stage5_result()
    saved = save_benchmark_run(_tiny_cpu_config(), result, base_dir=tmp_path)
    run_id = saved["run_id"]

    assert get_run_dir(run_id, base_dir=tmp_path) == Path(saved["run_dir"])

    metrics = load_run_metrics(run_id, base_dir=tmp_path)
    config = load_run_config(run_id, base_dir=tmp_path)
    raw_row = load_run_raw_row(run_id, base_dir=tmp_path)
    environment = load_run_environment(run_id, base_dir=tmp_path)

    assert metrics is not None
    assert metrics["stage"] == "stage5"

    assert config is not None
    assert config["device_requested"] == "cpu"

    assert raw_row is not None
    assert raw_row["best_final_path_name"] == "cache"

    assert environment is not None
    assert "python_version" in environment


def test_missing_or_unsafe_run_returns_none(tmp_path: Path) -> None:
    assert get_run_dir("missing-run", base_dir=tmp_path) is None
    assert get_run_dir("../bad", base_dir=tmp_path) is None
    assert load_run_metrics("missing-run", base_dir=tmp_path) is None


def test_list_runs_returns_newest_first(tmp_path: Path) -> None:
    first = save_benchmark_run(_tiny_cpu_config(), _sample_stage5_result(), base_dir=tmp_path)
    second = save_benchmark_run(_tiny_cpu_config(), _sample_stage5_result(), base_dir=tmp_path)

    first_dir = Path(first["run_dir"])
    second_dir = Path(second["run_dir"])

    first_dir.touch()
    second_dir.touch()

    runs = list_runs(limit=10, base_dir=tmp_path)

    assert len(runs) == 2
    assert runs[0]["run_id"] == second["run_id"]
    assert runs[1]["run_id"] == first["run_id"]
    assert runs[0]["stage"] == "stage5"


def test_list_runs_respects_limit(tmp_path: Path) -> None:
    save_benchmark_run(_tiny_cpu_config(), _sample_stage5_result(), base_dir=tmp_path)
    save_benchmark_run(_tiny_cpu_config(), _sample_stage5_result(), base_dir=tmp_path)

    runs = list_runs(limit=1, base_dir=tmp_path)

    assert len(runs) == 1