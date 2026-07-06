from fastapi import FastAPI, HTTPException
import torch

from app.benchmark_service import run_stage5_benchmark, run_stage6_benchmark
from app.schemas import Stage5BenchmarkRequest, Stage6BenchmarkRequest
from src.benchmark.run_store import (
    get_latest_run_dir,
    list_runs,
    load_latest_metrics,
    load_run_config,
    load_run_environment,
    load_run_metrics,
)

app = FastAPI(
    title="LLM Attention Benchmark API",
    version="0.2.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "llm-attention-benchmark-api",
    }


@app.get("/gpu-info")
def gpu_info():
    cuda_available = torch.cuda.is_available()

    info = {
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }

    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )

    return info


@app.get("/gpu")
def legacy_gpu_info():
    return gpu_info()


@app.post("/benchmark/stage5")
def benchmark_stage5(request: Stage5BenchmarkRequest):
    return run_stage5_benchmark(request)


@app.post("/benchmark/stage6")
def benchmark_stage6(request: Stage6BenchmarkRequest):
    return run_stage6_benchmark(request)

@app.get("/results/latest")
def latest_result():
    latest_run_dir = get_latest_run_dir()
    metrics = load_latest_metrics()

    if latest_run_dir is None or metrics is None:
        raise HTTPException(
            status_code=404,
            detail="No saved benchmark runs found",
        )

    metrics["run_id"] = latest_run_dir.name
    return metrics

@app.get("/results")
def results(limit: int = 20):
    return {
        "runs": list_runs(limit=limit),
    }


@app.get("/results/{run_id}")
def result_by_id(run_id: str):
    metrics = load_run_metrics(run_id)
    config = load_run_config(run_id)
    environment = load_run_environment(run_id)

    if metrics is None or config is None or environment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark run not found: {run_id}",
        )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "config": config,
        "environment": environment,
    }