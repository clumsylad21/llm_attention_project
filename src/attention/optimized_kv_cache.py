import torch


class OptimizedKVCache:
    """
    KV cache layout specialized for single-token decode.

    We store:
      - K in transposed form: [B, H, D, T]
      - V in normal form:     [B, H, T, D]

    Why?
    During decode, the hot operation is:
        scores = q @ K
    where q is [B, H, 1, D]

    If K is already stored as [B, H, D, T], we do not need to transpose
    K on every decode step.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.visible_len = 0

        # K stored transposed for decode-friendly matmul:
        # [B, H, D, capacity]
        self.k_cache_t = torch.empty(
            batch, heads, head_dim, capacity, device=device, dtype=dtype
        )

        # V stored normally:
        # [B, H, capacity, D]
        self.v_cache = torch.empty(
            batch, heads, capacity, head_dim, device=device, dtype=dtype
        )

    def append_prefix(self, k_prefix: torch.Tensor, v_prefix: torch.Tensor) -> None:
        """
        Append a whole prefix into the cache at once.

        Inputs:
          k_prefix: [B, H, T, D]
          v_prefix: [B, H, T, D]
        """
        if k_prefix.ndim != 4 or v_prefix.ndim != 4:
            raise ValueError("k_prefix and v_prefix must both be rank-4 tensors.")

        prefix_len = k_prefix.shape[2]

        if prefix_len > self.capacity:
            raise ValueError(
                f"Prefix length {prefix_len} exceeds cache capacity {self.capacity}."
            )

        # Store K transposed as [B, H, D, T]
        self.k_cache_t[:, :, :, :prefix_len] = k_prefix.transpose(-2, -1).contiguous()
        self.v_cache[:, :, :prefix_len, :] = v_prefix.contiguous()
        self.visible_len = prefix_len

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """
        Append one new token's K/V to the cache.

        Inputs:
          k_new: [B, H, 1, D]
          v_new: [B, H, 1, D]
        """
        if self.visible_len >= self.capacity:
            raise ValueError("KV cache is already full.")

        pos = self.visible_len

        # K stored as [B, H, D, 1]
        self.k_cache_t[:, :, :, pos : pos + 1] = k_new.transpose(-2, -1).contiguous()

        # V stored as [B, H, 1, D]
        self.v_cache[:, :, pos : pos + 1, :] = v_new.contiguous()

        self.visible_len += 1

    def get_visible_k_t(self) -> torch.Tensor:
        """
        Return visible K in transposed layout.

        Output:
          [B, H, D, visible_len]
        """
        return self.k_cache_t[:, :, :, : self.visible_len]

    def get_visible_v(self) -> torch.Tensor:
        """
        Return visible V.

        Output:
          [B, H, visible_len, D]
        """
        return self.v_cache[:, :, : self.visible_len, :]

    def capacity_bytes(self) -> int:
        """
        Return total allocated cache capacity in bytes.
        """
        return (
            self.k_cache_t.numel() + self.v_cache.numel()
        ) * self.k_cache_t.element_size()