from app.schemas import Stage5BenchmarkRequest, Stage6BenchmarkRequest
from src.benchmark.service import (
    run_stage5_benchmark as run_stage5_benchmark_service,
    run_stage6_benchmark as run_stage6_benchmark_service,
)


def run_stage5_benchmark(req: Stage5BenchmarkRequest) -> dict:
    result = run_stage5_benchmark_service(req.to_config())
    return result.to_dict(include_raw=False)


def run_stage6_benchmark(req: Stage6BenchmarkRequest) -> dict:
    result = run_stage6_benchmark_service(req.to_config())
    return result.to_dict(include_raw=False)