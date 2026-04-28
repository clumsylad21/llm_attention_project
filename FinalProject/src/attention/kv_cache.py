# src/attention/kv_cache.py

import torch


class KVCache:
    """
    Simple K/V cache for decoder-style attention.

    Internally stores:
        k_cache: [B, H, max_seq_len, D]
        v_cache: [B, H, max_seq_len, D]

    We keep a 'current_len' pointer telling us how much of the cache
    is currently filled with valid tokens.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Allocate full cache up front.
        # This is closer to how real inference engines work:
        # they allocate memory once and then fill it step by step.
        self.k_cache = torch.empty(
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.empty(
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        self.current_len = 0

    def reset(self) -> None:
        """
        Reset visible length to zero.
        The underlying memory is kept allocated.
        """
        self.current_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """
        Append newly computed K/V for one or more new tokens.

        Expected shapes:
            k_new: [B, H, T_new, D]
            v_new: [B, H, T_new, D]
        """
        if k_new.shape != v_new.shape:
            raise ValueError("k_new and v_new must have the same shape.")

        if k_new.dim() != 4:
            raise ValueError("k_new and v_new must have shape [B, H, T_new, D].")

        b, h, t_new, d = k_new.shape

        if b != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {b}.")
        if h != self.num_heads:
            raise ValueError(f"Expected num_heads {self.num_heads}, got {h}.")
        if d != self.head_dim:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {d}.")

        end = self.current_len + t_new
        if end > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: trying to store length {end}, "
                f"but max_seq_len is {self.max_seq_len}."
            )

        self.k_cache[:, :, self.current_len:end, :].copy_(k_new)
        self.v_cache[:, :, self.current_len:end, :].copy_(v_new)
        self.current_len = end

    def get_k(self) -> torch.Tensor:
        """
        Return only the currently visible part of K cache.
        Shape: [B, H, current_len, D]
        """
        return self.k_cache[:, :, :self.current_len, :]

    def get_v(self) -> torch.Tensor:
        """
        Return only the currently visible part of V cache.
        Shape: [B, H, current_len, D]
        """
        return self.v_cache[:, :, :self.current_len, :]

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return visible K and V together.
        """
        return self.get_k(), self.get_v()