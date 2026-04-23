# src/attention/stage6_cuda_extension.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

STAGE6_EXTENSION_BASE_NAME = "llm_attention_stage6_single_query_attention"
STAGE6_DEFAULT_TILE_TOKENS = 32

STAGE6_SUPPORTED_DTYPES = {
    torch.float16: "fp16",
    torch.float32: "fp32",
}


def _source_dir() -> Path:
    return Path(__file__).resolve().parent / "cuda"


def resolve_stage6_tile_tokens(tile_tokens: Optional[int] = None) -> int:
    """
    Resolve the tile size used when compiling the Stage 6 extension.

    We keep this configurable so you can do a tiny 16-vs-32 comparison
    without redesigning the benchmark framework.
    """
    if tile_tokens is None:
        raw_value = os.environ.get(
            "LLM_ATTENTION_STAGE6_TILE_TOKENS",
            str(STAGE6_DEFAULT_TILE_TOKENS),
        )
        try:
            tile_tokens = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                "LLM_ATTENTION_STAGE6_TILE_TOKENS must be an integer."
            ) from exc

    if tile_tokens <= 0:
        raise ValueError("Stage 6 tile size must be positive.")

    return tile_tokens


def _extension_name(tile_tokens: int) -> str:
    return f"{STAGE6_EXTENSION_BASE_NAME}_tile{tile_tokens}"


@lru_cache(maxsize=8)
def _load_stage6_extension(tile_tokens: int):
    source_dir = _source_dir()
    cpp_path = source_dir / "stage6_single_query_attention.cpp"
    cu_path = source_dir / "stage6_single_query_attention.cu"

    module_name = _extension_name(tile_tokens)

    try:
        module = load(
            name=module_name,
            sources=[str(cpp_path), str(cu_path)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-DSTAGE6_TILE_TOKENS={tile_tokens}",
            ],
            verbose=False,
        )
        return module, "loaded"
    except Exception as exc:
        return None, f"build_failed_{type(exc).__name__}"


def stage6_availability(
    device: torch.device,
    dtype: torch.dtype,
    enable_stage6: bool = True,
    tile_tokens: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Report whether the Stage 6 custom CUDA kernel is usable.
    """
    if not enable_stage6:
        return False, "stage6_disabled"

    if device.type != "cuda":
        return False, "cuda_only"

    if dtype not in STAGE6_SUPPORTED_DTYPES:
        return False, f"unsupported_dtype_{dtype}"

    if not torch.cuda.is_available():
        return False, "cuda_unavailable"

    try:
        resolved_tile_tokens = resolve_stage6_tile_tokens(tile_tokens)
    except ValueError as exc:
        return False, f"invalid_tile_tokens_{exc}"

    module, status = _load_stage6_extension(resolved_tile_tokens)
    if module is None:
        return False, status

    return True, status


def get_stage6_extension(
    device: torch.device,
    dtype: torch.dtype,
    enable_stage6: bool = True,
    tile_tokens: Optional[int] = None,
):
    """
    Return the loaded extension object or None, plus a status string.
    """
    available, status = stage6_availability(
        device=device,
        dtype=dtype,
        enable_stage6=enable_stage6,
        tile_tokens=tile_tokens,
    )
    if not available:
        return None, status

    resolved_tile_tokens = resolve_stage6_tile_tokens(tile_tokens)
    module, _ = _load_stage6_extension(resolved_tile_tokens)
    return module, status


def stage6_single_query_attention(
    q: torch.Tensor,
    k_visible_t: torch.Tensor,
    v_visible: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    tile_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Run the Stage 6 custom CUDA attention micro-kernel.

    Shapes:
        q:           [B, H, 1, D]
        k_visible_t: [B, H, D, T]
        v_visible:   [B, H, T, D]
        out:         [B, H, 1, D] if provided
    """
    module, status = get_stage6_extension(
        device=q.device,
        dtype=q.dtype,
        enable_stage6=True,
        tile_tokens=tile_tokens,
    )
    if module is None:
        raise RuntimeError(f"Stage 6 extension unavailable: {status}")

    if q.size(2) != 1:
        raise ValueError("Stage 6 attention expects q to have shape [B, H, 1, D].")

    q_contig = q.contiguous()
    k_contig = k_visible_t.contiguous()
    v_contig = v_visible.contiguous()

    if out is None:
        return module.forward(q_contig, k_contig, v_contig)

    if out.shape != q.shape:
        raise ValueError(
            f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}."
        )
    if out.device != q.device:
        raise ValueError("out must be on the same device as q.")
    if out.dtype != q.dtype:
        raise ValueError("out must have the same dtype as q.")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous.")

    module.forward_out(q_contig, k_contig, v_contig, out)
    return out
