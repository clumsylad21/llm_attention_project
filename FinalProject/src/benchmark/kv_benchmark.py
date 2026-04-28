import time

import torch

from src.attention.decode import (
    naive_decode_step,
    kv_cache_decode_step,
    prefill_kv_cache,
)
from src.attention.kv_cache import KVCache


def sync_if_needed(device: torch.device) -> None:
    """
    Synchronize only for CUDA.

    CUDA kernels run asynchronously, so without synchronization
    the timing would be incorrect.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_ms(values_ms: list[float]) -> dict[str, float]:
    """
    Convert a list of timing samples into mean/std/min/max stats.
    """
    if len(values_ms) == 0:
        raise ValueError("values_ms is empty.")

    t = torch.tensor(values_ms, dtype=torch.float64)

    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std(unbiased=False).item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
    }


def benchmark_naive_per_step(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    weights,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark naive decode one step at a time.

    We time each generated step separately, so this metric answers:
    "How expensive is one generated token?"
    """
    step_latencies_ms: list[float] = []

    for run_idx in range(warmup + iters):
        for step_idx in range(gen_steps):
            visible_len = prompt_len + step_idx + 1

            sync_if_needed(device)
            t0 = time.perf_counter()

            _ = naive_decode_step(
                all_hidden_states,
                weights,
                num_heads,
                head_dim,
                visible_len,
            )

            sync_if_needed(device)
            t1 = time.perf_counter()

            if run_idx >= warmup:
                step_latencies_ms.append((t1 - t0) * 1000.0)

    return summarize_ms(step_latencies_ms)


def benchmark_naive_total(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    weights,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark the full naive decode loop.

    This metric answers:
    "How long does the whole generated segment take?"
    """
    total_latencies_ms: list[float] = []

    for run_idx in range(warmup + iters):
        sync_if_needed(device)
        t0 = time.perf_counter()

        for step_idx in range(gen_steps):
            visible_len = prompt_len + step_idx + 1
            _ = naive_decode_step(
                all_hidden_states,
                weights,
                num_heads,
                head_dim,
                visible_len,
            )

        sync_if_needed(device)
        t1 = time.perf_counter()

        if run_idx >= warmup:
            total_latencies_ms.append((t1 - t0) * 1000.0)

    return summarize_ms(total_latencies_ms)


def benchmark_cache_prefill(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    weights,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark prompt prefill for the KV cache.

    This is stored separately because prefill is prompt processing work,
    not generated-token decode work.
    """
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]

    prefill_latencies_ms: list[float] = []

    for run_idx in range(warmup + iters):
        cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=total_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=all_hidden_states.dtype,
        )

        sync_if_needed(device)
        t0 = time.perf_counter()

        prefill_kv_cache(
            prompt_hidden_states,
            weights,
            num_heads,
            head_dim,
            cache,
        )

        sync_if_needed(device)
        t1 = time.perf_counter()

        if run_idx >= warmup:
            prefill_latencies_ms.append((t1 - t0) * 1000.0)

    return summarize_ms(prefill_latencies_ms)


def benchmark_cache_per_step(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    weights,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark KV-cache decode one step at a time.

    Prefill is intentionally excluded here because this metric is meant
    to reflect generated-token latency only.
    """
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len:prompt_len + gen_steps, :]

    step_latencies_ms: list[float] = []

    for run_idx in range(warmup + iters):
        cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=total_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=all_hidden_states.dtype,
        )

        prefill_kv_cache(
            prompt_hidden_states,
            weights,
            num_heads,
            head_dim,
            cache,
        )

        for step_idx in range(gen_steps):
            current_hidden_state = generated_hidden_states[:, step_idx:step_idx + 1, :]

            sync_if_needed(device)
            t0 = time.perf_counter()

            _ = kv_cache_decode_step(
                current_hidden_state,
                weights,
                num_heads,
                head_dim,
                cache,
            )

            sync_if_needed(device)
            t1 = time.perf_counter()

            if run_idx >= warmup:
                step_latencies_ms.append((t1 - t0) * 1000.0)

    return summarize_ms(step_latencies_ms)


def benchmark_cache_total(
    all_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    weights,
    num_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Benchmark the full KV-cache decode loop for generated tokens only.

    Important:
    prompt prefill is excluded from this metric and recorded separately.
    """
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len:prompt_len + gen_steps, :]

    total_latencies_ms: list[float] = []

    for run_idx in range(warmup + iters):
        cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=total_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=all_hidden_states.dtype,
        )

        prefill_kv_cache(
            prompt_hidden_states,
            weights,
            num_heads,
            head_dim,
            cache,
        )

        sync_if_needed(device)
        t0 = time.perf_counter()

        for step_idx in range(gen_steps):
            current_hidden_state = generated_hidden_states[:, step_idx:step_idx + 1, :]
            _ = kv_cache_decode_step(
                current_hidden_state,
                weights,
                num_heads,
                head_dim,
                cache,
            )

        sync_if_needed(device)
        t1 = time.perf_counter()

        if run_idx >= warmup:
            total_latencies_ms.append((t1 - t0) * 1000.0)

    return summarize_ms(total_latencies_ms)