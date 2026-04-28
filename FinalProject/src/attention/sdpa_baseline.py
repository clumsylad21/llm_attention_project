import torch
import torch.nn.functional as F

from src.common.seed import set_seed


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def tensor_mib(t: torch.Tensor) -> float:
    return bytes_to_mib(t.numel() * t.element_size())


@torch.inference_mode()
def run_prefill_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Full causal self-attention over the whole prompt.

    q, k, v shape: [B, H, S, D]
    """
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
    return out


@torch.inference_mode()
def run_decode_step_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Single-token decode-style attention.

    q shape: [B, H, 1, D]
    k, v shape: [B, H, S, D]

    Decode step me k/v already sirf visible context hota hai,
    isliye yahan future masking ki zarurat nahi.
    """
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    return out


def make_prefill_tensors(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
):
    set_seed(seed)

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    return q.contiguous(), k.contiguous(), v.contiguous()


def make_decode_tensors(
    batch: int,
    heads: int,
    context_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
):
    set_seed(seed)

    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, context_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, context_len, head_dim, device=device, dtype=dtype)

    return q.contiguous(), k.contiguous(), v.contiguous()