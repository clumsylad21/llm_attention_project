from app.schemas import Stage5BenchmarkRequest, Stage6BenchmarkRequest
from src.benchmark.run_store import save_benchmark_run
from src.benchmark.service import (
    run_stage5_benchmark as run_stage5_benchmark_service,
    run_stage6_benchmark as run_stage6_benchmark_service,
)


def run_stage5_benchmark(req: Stage5BenchmarkRequest) -> dict:
    config = req.to_config()
    result = run_stage5_benchmark_service(config)

    saved = save_benchmark_run(config, result)

    response = result.to_dict(include_raw=False)
    response["run_id"] = saved["run_id"]
    return response


def run_stage6_benchmark(req: Stage6BenchmarkRequest) -> dict:
    config = req.to_config()
    result = run_stage6_benchmark_service(config)

    saved = save_benchmark_run(config, result)

    response = result.to_dict(include_raw=False)
    response["run_id"] = saved["run_id"]
    return response