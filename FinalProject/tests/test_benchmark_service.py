from src.benchmark.config import BenchmarkConfig
from src.benchmark.result import BenchmarkResult
from src.benchmark.service import run_stage5_benchmark, run_stage6_benchmark


def tiny_cpu_config() -> BenchmarkConfig:
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
        seed=0,
        enable_compile=False,
        enable_cuda_graphs=False,
        enable_stage6=False,
    )


def test_run_stage5_benchmark_tiny_cpu_smoke():
    result = run_stage5_benchmark(tiny_cpu_config())

    assert isinstance(result, BenchmarkResult)
    assert result.stage == "stage5"
    assert result.device == "cpu"
    assert result.all_correct is True
    assert result.best_backend != "unknown"
    assert result.best_mean_ms >= 0.0

    backend_names = {backend.name for backend in result.backends}
    assert {"naive", "cache", "stage4a", "compiled", "cuda_graph"} <= backend_names


def test_run_stage6_benchmark_tiny_cpu_smoke():
    result = run_stage6_benchmark(tiny_cpu_config())

    assert isinstance(result, BenchmarkResult)
    assert result.stage == "stage6"
    assert result.device == "cpu"
    assert result.all_correct is True
    assert result.best_backend != "unknown"
    assert result.best_mean_ms >= 0.0

    backend_names = {backend.name for backend in result.backends}
    assert "stage6_custom_cuda" in backend_names

    stage6_backend = next(
        backend for backend in result.backends if backend.name == "stage6_custom_cuda"
    )
    assert stage6_backend.available is False
    assert stage6_backend.status == "stage6_disabled"


def test_benchmark_result_can_be_serialized_without_raw():
    result = run_stage5_benchmark(tiny_cpu_config())

    payload = result.to_dict(include_raw=False)

    assert payload["stage"] == "stage5"
    assert "raw" not in payload
    assert isinstance(payload["backends"], list)