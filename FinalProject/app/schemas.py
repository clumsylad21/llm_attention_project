from typing import Literal

from pydantic import BaseModel, Field

from src.benchmark.config import BenchmarkConfig


class BenchmarkRequest(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    dtype: Literal["fp16", "bf16", "fp32"] = "fp32"
    batch: int = Field(default=1, ge=1, le=4)
    heads: int = Field(default=8, ge=1, le=32)
    head_dim: int = Field(default=64, ge=16, le=256)
    prompt_len: int = Field(default=64, ge=8, le=4096)
    gen_steps: int = Field(default=4, ge=1, le=128)
    warmup: int = Field(default=1, ge=0, le=50)
    iters: int = Field(default=2, ge=1, le=200)
    seed: int = 0
    compile_mode: str = "reduce-overhead"
    fullgraph: bool = False
    enable_compile: bool = False
    enable_cuda_graphs: bool = False
    enable_stage6: bool = False

    def to_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            device_requested=self.device,
            dtype_name=self.dtype,
            batch=self.batch,
            heads=self.heads,
            head_dim=self.head_dim,
            prompt_len=self.prompt_len,
            gen_steps=self.gen_steps,
            warmup=self.warmup,
            iters=self.iters,
            seed=self.seed,
            compile_mode=self.compile_mode,
            fullgraph=self.fullgraph,
            enable_compile=self.enable_compile,
            enable_cuda_graphs=self.enable_cuda_graphs,
            enable_stage6=self.enable_stage6,
        )


class Stage5BenchmarkRequest(BenchmarkRequest):
    pass


class Stage6BenchmarkRequest(BenchmarkRequest):
    pass