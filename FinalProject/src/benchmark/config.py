from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


_SUPPORTED_DTYPES = {"fp16", "bf16", "fp32"}


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")

    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")

    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Unified internal benchmark config for Stage 5/Stage 6 decode benchmarks.

    The field names intentionally match the existing benchmark builders so this
    config can be passed through without translating the hot-path arguments.
    """

    device_requested: str = "auto"
    dtype_name: str = "fp32"
    batch: int = 1
    heads: int = 8
    head_dim: int = 64
    prompt_len: int = 512
    gen_steps: int = 32
    warmup: int = 10
    iters: int = 50
    seed: int = 0
    compile_mode: str = "reduce-overhead"
    fullgraph: bool = False
    enable_compile: bool = True
    enable_cuda_graphs: bool = True
    enable_stage6: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.device_requested, str) or not self.device_requested:
            raise ValueError("device_requested must be a non-empty string")

        if self.dtype_name not in _SUPPORTED_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_DTYPES))
            raise ValueError(
                f"Unsupported dtype_name '{self.dtype_name}'. Choose from: {supported}"
            )

        _validate_positive_int("batch", self.batch)
        _validate_positive_int("heads", self.heads)
        _validate_positive_int("head_dim", self.head_dim)
        _validate_positive_int("prompt_len", self.prompt_len)
        _validate_positive_int("gen_steps", self.gen_steps)
        _validate_non_negative_int("warmup", self.warmup)
        _validate_positive_int("iters", self.iters)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")

        if not isinstance(self.compile_mode, str) or not self.compile_mode:
            raise ValueError("compile_mode must be a non-empty string")

        for name in (
            "fullgraph",
            "enable_compile",
            "enable_cuda_graphs",
            "enable_stage6",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BenchmarkConfig":
        """
        Build a config from dict-like inputs such as JSON payloads.
        """
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_stage5_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("enable_stage6")
        return kwargs

    def to_stage6_kwargs(self) -> dict[str, Any]:
        return asdict(self)
