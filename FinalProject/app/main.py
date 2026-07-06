from fastapi import FastAPI
import torch

from app.benchmark_service import run_stage5_benchmark, run_stage6_benchmark
from app.schemas import Stage5BenchmarkRequest, Stage6BenchmarkRequest

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