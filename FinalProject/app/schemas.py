from typing import Literal
from pydantic import BaseModel, Field


class Stage5BenchmarkRequest(BaseModel):
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
    enable_compile: bool = False
    enable_cuda_graphs: bool = False