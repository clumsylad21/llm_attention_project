# src/attention/stage6_cuda_extension.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

STAGE6_EXTENSION_NAME = "llm_attention_stage6_single_query_attention"
STAGE6_TILE_TOKENS = 16
STAGE6_SUPPORTED_DTYPES = {
    torch.float16: "fp16",
    torch.float32: "fp32",
}


def _source_dir() -> Path:
    return Path(__file__).resolve().parent / "cuda"


@lru_cache(maxsize=1)
def _load_stage6_extension():
    source_dir = _source_dir()
    cpp_path = source_dir / "stage6_single_query_attention.cpp"
    cu_path = source_dir / "stage6_single_query_attention.cu"

    try:
        module = load(
            name=STAGE6_EXTENSION_NAME,
            sources=[str(cpp_path), str(cu_path)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return module, "loaded"
    except Exception as exc:
        return None, f"build_failed_{type(exc).__name__}"


def stage6_availability(
    device: torch.device,
    dtype: torch.dtype,
    enable_stage6: bool = True,
) -> tuple[bool, str]:
    """
    Report whether the Stage 6 custom CUDA kernel is usable.

    We keep this separate from the runner so benchmark code can decide
    whether to time the real custom kernel or record it as unavailable.
    """
    if not enable_stage6:
        return False, "stage6_disabled"

    if device.type != "cuda":
        return False, "cuda_only"

    if dtype not in STAGE6_SUPPORTED_DTYPES:
        return False, f"unsupported_dtype_{dtype}"

    if not torch.cuda.is_available():
        return False, "cuda_unavailable"

    module, status = _load_stage6_extension()
    if module is None:
        return False, status

    return True, status


def get_stage6_extension(
    device: torch.device,
    dtype: torch.dtype,
    enable_stage6: bool = True,
):
    """
    Return the loaded extension object or None, plus a status string.
    """
    available, status = stage6_availability(
        device=device,
        dtype=dtype,
        enable_stage6=enable_stage6,
    )
    if not available:
        return None, status

    module, _ = _load_stage6_extension()
    return module, status


def stage6_single_query_attention(
    q: torch.Tensor,
    k_visible_t: torch.Tensor,
    v_visible: torch.Tensor,
) -> torch.Tensor:
    """
    Run the Stage 6 custom CUDA attention micro-kernel.

    Shapes:
        q:           [B, H, 1, D]
        k_visible_t: [B, H, D, T]
        v_visible:   [B, H, T, D]

    Notes:
    - This is forward-only and decode-only.
    - The extension implements only the attention core.
    - Projection, KV-cache bookkeeping, and experiment orchestration
      stay in Python/PyTorch.
    """
    module, status = get_stage6_extension(
        device=q.device,
        dtype=q.dtype,
        enable_stage6=True,
    )

    if module is None:
        raise RuntimeError(f"Stage 6 extension unavailable: {status}")

    if q.size(2) != 1:
        raise ValueError("Stage 6 attention expects q to have shape [B, H, 1, D].")

    return module.forward(
        q.contiguous(),
        k_visible_t.contiguous(),
        v_visible.contiguous(),
    )
