# run_kv_cache.py

import argparse
import csv
import time

import torch

from src.attention.decode import (
    create_projection_weights,
    naive_decode_step,
    kv_cache_decode_step,
    prefill_kv_cache,
    run_naive_decode,
    run_kv_cache_decode,
)
from src.attention.kv_cache import KVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: benchmark naive decode vs KV-cache decode."
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)

    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-steps", type=int, default=64)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--csv-out", type=str, default=None)

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")

    return torch.device(device_str)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    dtype = mapping[dtype_str]

    # CPU fp16 matmul is often a bad experience or unsupported depending on build.
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 on CPU is not recommended here. Use fp32/bf16 or switch to CUDA.")

    return dtype


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def summarize_ms(values_ms: list[float]) -> dict:
    t = torch.tensor(values_ms, dtype=torch.float64)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std(unbiased=False).item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
    }


def get_compare_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Relax tolerances a bit for lower precision types.
    if dtype == torch.float32:
        return 1e-5, 1e-4
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    return 1e-5, 1e-4


def build_synthetic_inputs(
    batch: int,
    prompt_len: int,
    gen_steps: int,
    heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    """
    Build synthetic decoder hidden states.

    model_dim = heads * head_dim
    total sequence = prompt + generated tokens

    Shape returned:
        hidden_states: [B, total_seq_len, model_dim]
    """
    model_dim = heads * head_dim
    total_seq_len = prompt_len + gen_steps

    hidden_states = torch.randn(
        batch,
        total_seq_len,
        model_dim,
        device=device,
        dtype=dtype,
    )

    return hidden_states, model_dim


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
) -> dict:
    step_latencies_ms = []

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
) -> dict:
    total_latencies_ms = []

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
) -> dict:
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len:prompt_len + gen_steps, :]

    step_latencies_ms = []

    for run_idx in range(warmup + iters):
        cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=total_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=all_hidden_states.dtype,
        )

        # We treat prompt prefill as setup, not as decode-step cost.
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
) -> dict:
    batch_size = all_hidden_states.size(0)
    total_seq_len = all_hidden_states.size(1)

    prompt_hidden_states = all_hidden_states[:, :prompt_len, :]
    generated_hidden_states = all_hidden_states[:, prompt_len:prompt_len + gen_steps, :]

    total_latencies_ms = []

    for run_idx in range(warmup + iters):
        cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=total_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=all_hidden_states.dtype,
        )

        # Prompt prefill excluded from decode total.
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


def maybe_write_csv(
    csv_path: str,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    heads: int,
    head_dim: int,
    prompt_len: int,
    gen_steps: int,
    naive_step_stats: dict,
    naive_total_stats: dict,
    cache_step_stats: dict,
    cache_total_stats: dict,
) -> None:
    fieldnames = [
        "mode",
        "device",
        "dtype",
        "batch",
        "heads",
        "head_dim",
        "prompt_len",
        "gen_steps",
        "step_mean_ms",
        "step_std_ms",
        "step_min_ms",
        "step_max_ms",
        "total_mean_ms",
        "total_std_ms",
        "total_min_ms",
        "total_max_ms",
    ]

    rows = [
        {
            "mode": "naive",
            "device": str(device),
            "dtype": str(dtype),
            "batch": batch,
            "heads": heads,
            "head_dim": head_dim,
            "prompt_len": prompt_len,
            "gen_steps": gen_steps,
            "step_mean_ms": naive_step_stats["mean_ms"],
            "step_std_ms": naive_step_stats["std_ms"],
            "step_min_ms": naive_step_stats["min_ms"],
            "step_max_ms": naive_step_stats["max_ms"],
            "total_mean_ms": naive_total_stats["mean_ms"],
            "total_std_ms": naive_total_stats["std_ms"],
            "total_min_ms": naive_total_stats["min_ms"],
            "total_max_ms": naive_total_stats["max_ms"],
        },
        {
            "mode": "kv_cache",
            "device": str(device),
            "dtype": str(dtype),
            "batch": batch,
            "heads": heads,
            "head_dim": head_dim,
            "prompt_len": prompt_len,
            "gen_steps": gen_steps,
            "step_mean_ms": cache_step_stats["mean_ms"],
            "step_std_ms": cache_step_stats["std_ms"],
            "step_min_ms": cache_step_stats["min_ms"],
            "step_max_ms": cache_step_stats["max_ms"],
            "total_mean_ms": cache_total_stats["mean_ms"],
            "total_std_ms": cache_total_stats["std_ms"],
            "total_min_ms": cache_total_stats["min_ms"],
            "total_max_ms": cache_total_stats["max_ms"],
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    set_seed(args.seed)

    hidden_states, model_dim = build_synthetic_inputs(
        batch=args.batch,
        prompt_len=args.prompt_len,
        gen_steps=args.gen_steps,
        heads=args.heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
    )

    weights = create_projection_weights(
        model_dim=model_dim,
        device=device,
        dtype=dtype,
    )

    # -----------------------------
    # Correctness / numerical check
    # -----------------------------
    naive_outputs = run_naive_decode(
        hidden_states,
        args.prompt_len,
        args.gen_steps,
        weights,
        args.heads,
        args.head_dim,
    )

    cache_outputs = run_kv_cache_decode(
        hidden_states,
        args.prompt_len,
        args.gen_steps,
        weights,
        args.heads,
        args.head_dim,
    )

    atol, rtol = get_compare_tolerances(dtype)

    diff = (naive_outputs.float() - cache_outputs.float()).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    outputs_close = torch.allclose(
        naive_outputs.float(),
        cache_outputs.float(),
        atol=atol,
        rtol=rtol,
    )

    naive_checksum = naive_outputs.float().mean().item()
    cache_checksum = cache_outputs.float().mean().item()

    # -----------------------------
    # Benchmark
    # -----------------------------
    naive_step_stats = benchmark_naive_per_step(
        hidden_states,
        args.prompt_len,
        args.gen_steps,
        weights,
        args.heads,
        args.head_dim,
        args.warmup,
        args.iters,
        device,
    )

    naive_total_stats = benchmark_naive_total(
        hidden_states,
        args.prompt_len,
        args.gen_steps,
        weights,
        args.heads,
        args.head_dim,
        args.warmup,
        args.iters,
        device,
    )

    cache_step_stats = benchmark_cache_per_step(
        hidden_states,
        args.prompt_len,
        args.gen_steps,
        weights,
        args.heads,
        args.head_dim,
        args.warmup,
        args.iters,
        device,
    )

    cache_total_stats = benchmark_cache_total(
        hidden_states,
        args.prompt_len,
        args.gen_steps,
        weights,
        args.heads,
        args.head_dim,
        args.warmup,
        args.iters,
        device,
    )

    # -----------------------------
    # Memory reporting
    # -----------------------------
    hidden_states_mib = bytes_to_mib(tensor_nbytes(hidden_states))
    weights_mib = bytes_to_mib(
        tensor_nbytes(weights.w_q) +
        tensor_nbytes(weights.w_k) +
        tensor_nbytes(weights.w_v)
    )

    cache_capacity_bytes = (
        2
        * args.batch
        * args.heads
        * (args.prompt_len + args.gen_steps)
        * args.head_dim
        * hidden_states.element_size()
    )
    cache_capacity_mib = bytes_to_mib(cache_capacity_bytes)

    # -----------------------------
    # Print report
    # -----------------------------
    print("=" * 95)
    print("Stage 2 - Naive Decode vs KV Cache")
    print("=" * 95)
    print(f"device          : {device}")
    print(f"dtype           : {dtype}")
    print(f"batch           : {args.batch}")
    print(f"heads           : {args.heads}")
    print(f"head_dim        : {args.head_dim}")
    print(f"model_dim       : {model_dim}")
    print(f"prompt_len      : {args.prompt_len}")
    print(f"gen_steps       : {args.gen_steps}")
    print(f"warmup          : {args.warmup}")
    print(f"iters           : {args.iters}")
    print("=" * 95)

    print("Numerical check")
    print(f"allclose        : {outputs_close}  (atol={atol}, rtol={rtol})")
    print(f"max_abs_diff    : {max_abs_diff:.8f}")
    print(f"mean_abs_diff   : {mean_abs_diff:.8f}")
    print(f"naive_checksum  : {naive_checksum:.8f}")
    print(f"cache_checksum  : {cache_checksum:.8f}")
    print("-" * 95)

    print("Memory")
    print(f"hidden_states   : {hidden_states_mib:.4f} MiB")
    print(f"weights (QKV)   : {weights_mib:.4f} MiB")
    print(f"kv_cache_cap    : {cache_capacity_mib:.4f} MiB")
    print("-" * 95)

    print("Naive decode")
    print(
        f"per-step        : mean={naive_step_stats['mean_ms']:8.4f} ms   "
        f"std={naive_step_stats['std_ms']:8.4f}   "
        f"min={naive_step_stats['min_ms']:8.4f}   "
        f"max={naive_step_stats['max_ms']:8.4f}"
    )
    print(
        f"total           : mean={naive_total_stats['mean_ms']:8.4f} ms   "
        f"std={naive_total_stats['std_ms']:8.4f}   "
        f"min={naive_total_stats['min_ms']:8.4f}   "
        f"max={naive_total_stats['max_ms']:8.4f}"
    )
    print("-" * 95)

    print("KV-cache decode")
    print(
        f"per-step        : mean={cache_step_stats['mean_ms']:8.4f} ms   "
        f"std={cache_step_stats['std_ms']:8.4f}   "
        f"min={cache_step_stats['min_ms']:8.4f}   "
        f"max={cache_step_stats['max_ms']:8.4f}"
    )
    print(
        f"total           : mean={cache_total_stats['mean_ms']:8.4f} ms   "
        f"std={cache_total_stats['std_ms']:8.4f}   "
        f"min={cache_total_stats['min_ms']:8.4f}   "
        f"max={cache_total_stats['max_ms']:8.4f}"
    )
    print("-" * 95)

    print("Speedup")
    print(
        f"per-step speedup: {naive_step_stats['mean_ms'] / cache_step_stats['mean_ms']:.4f}x"
    )
    print(
        f"total speedup   : {naive_total_stats['mean_ms'] / cache_total_stats['mean_ms']:.4f}x"
    )

    if args.csv_out is not None:
        maybe_write_csv(
            args.csv_out,
            device,
            dtype,
            args.batch,
            args.heads,
            args.head_dim,
            args.prompt_len,
            args.gen_steps,
            naive_step_stats,
            naive_total_stats,
            cache_step_stats,
            cache_total_stats,
        )
        print("-" * 95)
        print(f"CSV written to  : {args.csv_out}")


if __name__ == "__main__":
    main()