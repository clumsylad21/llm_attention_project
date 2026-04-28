import torch

from src.attention.decode import (
    create_projection_weights,
    run_naive_decode,
    run_kv_cache_decode,
)
from src.benchmark.kv_benchmark import (
    benchmark_naive_per_step,
    benchmark_naive_total,
    benchmark_cache_prefill,
    benchmark_cache_per_step,
    benchmark_cache_total,
)


def resolve_device(device_str: str) -> torch.device:
    """
    Resolve the requested device string to a torch.device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")

    return torch.device(device_str)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """
    Resolve dtype string to torch dtype.

    We keep the CPU fp16 guard because CPU fp16 matmul can be a bad experience
    or unsupported depending on build/platform.
    """
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    dtype = mapping[dtype_str]

    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 on CPU is not recommended here. Use fp32/bf16 or switch to CUDA.")

    return dtype


def set_seed(seed: int, device: torch.device) -> None:
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def get_compare_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """
    Numerical comparison tolerances.

    Lower precision types need looser tolerances.
    """
    if dtype == torch.float32:
        return 1e-5, 1e-4
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    return 1e-5, 1e-4


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """
    Return number of bytes occupied by a tensor.
    """
    return tensor.numel() * tensor.element_size()


def bytes_to_mib(num_bytes: int) -> float:
    """
    Convert bytes to MiB.
    """
    return num_bytes / (1024.0 * 1024.0)


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

    hidden_states shape:
        [B, total_seq_len, model_dim]

    where:
        model_dim = heads * head_dim
        total_seq_len = prompt_len + gen_steps
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


@torch.no_grad()
def run_single_kv_experiment(
    device_str: str,
    dtype_str: str,
    batch: int,
    heads: int,
    head_dim: int,
    prompt_len: int,
    gen_steps: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict:
    """
    Run one complete naive-vs-KV-cache experiment.

    This is the central Stage 3 function.

    It:
    1. resolves device/dtype
    2. builds synthetic hidden states + projection weights
    3. runs correctness check
    4. runs timing benchmarks
    5. computes memory numbers
    6. returns one flat dictionary ready for CSV
    """
    device = resolve_device(device_str)
    dtype = resolve_dtype(dtype_str, device)

    set_seed(seed, device)

    hidden_states, model_dim = build_synthetic_inputs(
        batch=batch,
        prompt_len=prompt_len,
        gen_steps=gen_steps,
        heads=heads,
        head_dim=head_dim,
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
        prompt_len,
        gen_steps,
        weights,
        heads,
        head_dim,
    )

    cache_outputs = run_kv_cache_decode(
        hidden_states,
        prompt_len,
        gen_steps,
        weights,
        heads,
        head_dim,
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
        prompt_len,
        gen_steps,
        weights,
        heads,
        head_dim,
        warmup,
        iters,
        device,
    )

    naive_total_stats = benchmark_naive_total(
        hidden_states,
        prompt_len,
        gen_steps,
        weights,
        heads,
        head_dim,
        warmup,
        iters,
        device,
    )

    cache_prefill_stats = benchmark_cache_prefill(
        hidden_states,
        prompt_len,
        weights,
        heads,
        head_dim,
        warmup,
        iters,
        device,
    )

    cache_step_stats = benchmark_cache_per_step(
        hidden_states,
        prompt_len,
        gen_steps,
        weights,
        heads,
        head_dim,
        warmup,
        iters,
        device,
    )

    cache_total_stats = benchmark_cache_total(
        hidden_states,
        prompt_len,
        gen_steps,
        weights,
        heads,
        head_dim,
        warmup,
        iters,
        device,
    )

    # -----------------------------
    # Memory reporting
    # -----------------------------
    hidden_states_bytes = tensor_nbytes(hidden_states)

    weights_bytes = (
        tensor_nbytes(weights.w_q)
        + tensor_nbytes(weights.w_k)
        + tensor_nbytes(weights.w_v)
    )

    kv_cache_capacity_bytes = (
        2
        * batch
        * heads
        * (prompt_len + gen_steps)
        * head_dim
        * hidden_states.element_size()
    )

    step_speedup = naive_step_stats["mean_ms"] / cache_step_stats["mean_ms"]
    total_speedup = naive_total_stats["mean_ms"] / cache_total_stats["mean_ms"]

    row = {
        # status
        "status": "ok",
        "error_message": "",

        # config
        "device_requested": device_str,
        "resolved_device": str(device),
        "dtype_name": dtype_str,
        "resolved_dtype": str(dtype),
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "model_dim": model_dim,
        "prompt_len": prompt_len,
        "gen_steps": gen_steps,
        "total_seq_len": prompt_len + gen_steps,
        "warmup": warmup,
        "iters": iters,
        "seed": seed,

        # correctness
        "allclose": outputs_close,
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "naive_checksum": naive_checksum,
        "cache_checksum": cache_checksum,

        # memory
        "hidden_states_bytes": hidden_states_bytes,
        "hidden_states_mib": bytes_to_mib(hidden_states_bytes),
        "weights_bytes": weights_bytes,
        "weights_mib": bytes_to_mib(weights_bytes),
        "kv_cache_capacity_bytes": kv_cache_capacity_bytes,
        "kv_cache_capacity_mib": bytes_to_mib(kv_cache_capacity_bytes),

        # naive per-step
        "naive_decode_step_mean_ms": naive_step_stats["mean_ms"],
        "naive_decode_step_std_ms": naive_step_stats["std_ms"],
        "naive_decode_step_min_ms": naive_step_stats["min_ms"],
        "naive_decode_step_max_ms": naive_step_stats["max_ms"],

        # naive total
        "naive_decode_total_mean_ms": naive_total_stats["mean_ms"],
        "naive_decode_total_std_ms": naive_total_stats["std_ms"],
        "naive_decode_total_min_ms": naive_total_stats["min_ms"],
        "naive_decode_total_max_ms": naive_total_stats["max_ms"],

        # cache prefill
        "cache_prefill_mean_ms": cache_prefill_stats["mean_ms"],
        "cache_prefill_std_ms": cache_prefill_stats["std_ms"],
        "cache_prefill_min_ms": cache_prefill_stats["min_ms"],
        "cache_prefill_max_ms": cache_prefill_stats["max_ms"],

        # cache per-step
        "cache_decode_step_mean_ms": cache_step_stats["mean_ms"],
        "cache_decode_step_std_ms": cache_step_stats["std_ms"],
        "cache_decode_step_min_ms": cache_step_stats["min_ms"],
        "cache_decode_step_max_ms": cache_step_stats["max_ms"],

        # cache total
        "cache_decode_total_mean_ms": cache_total_stats["mean_ms"],
        "cache_decode_total_std_ms": cache_total_stats["std_ms"],
        "cache_decode_total_min_ms": cache_total_stats["min_ms"],
        "cache_decode_total_max_ms": cache_total_stats["max_ms"],

        # speedups
        "step_speedup": step_speedup,
        "total_speedup": total_speedup,
    }

    return row


def build_error_row(
    device_str: str,
    dtype_str: str,
    batch: int,
    heads: int,
    head_dim: int,
    prompt_len: int,
    gen_steps: int,
    warmup: int,
    iters: int,
    seed: int,
    error_message: str,
) -> dict:
    """
    Build a minimal row for a failed sweep experiment.
    """
    return {
        "status": "error",
        "error_message": error_message,
        "device_requested": device_str,
        "dtype_name": dtype_str,
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "prompt_len": prompt_len,
        "gen_steps": gen_steps,
        "warmup": warmup,
        "iters": iters,
        "seed": seed,
    }