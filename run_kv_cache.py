import argparse

from src.benchmark.csv_utils import write_rows_to_csv
from src.benchmark.kv_experiment import run_single_kv_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: single naive-vs-KV-cache experiment."
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


def main() -> None:
    args = parse_args()

    row = run_single_kv_experiment(
        device_str=args.device,
        dtype_str=args.dtype,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        prompt_len=args.prompt_len,
        gen_steps=args.gen_steps,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )

    print("=" * 100)
    print("Stage 3 - Naive Decode vs KV Cache")
    print("=" * 100)
    print(f"device requested     : {row['device_requested']}")
    print(f"resolved device      : {row['resolved_device']}")
    print(f"dtype name           : {row['dtype_name']}")
    print(f"resolved dtype       : {row['resolved_dtype']}")
    print(f"batch                : {row['batch']}")
    print(f"heads                : {row['heads']}")
    print(f"head_dim             : {row['head_dim']}")
    print(f"model_dim            : {row['model_dim']}")
    print(f"prompt_len           : {row['prompt_len']}")
    print(f"gen_steps            : {row['gen_steps']}")
    print(f"total_seq_len        : {row['total_seq_len']}")
    print(f"warmup               : {row['warmup']}")
    print(f"iters                : {row['iters']}")
    print("=" * 100)

    print("Numerical check")
    print(f"allclose             : {row['allclose']}  (atol={row['atol']}, rtol={row['rtol']})")
    print(f"max_abs_diff         : {row['max_abs_diff']:.8f}")
    print(f"mean_abs_diff        : {row['mean_abs_diff']:.8f}")
    print(f"naive_checksum       : {row['naive_checksum']:.8f}")
    print(f"cache_checksum       : {row['cache_checksum']:.8f}")
    print("-" * 100)

    print("Memory")
    print(f"hidden_states        : {row['hidden_states_mib']:.4f} MiB")
    print(f"weights (QKV)        : {row['weights_mib']:.4f} MiB")
    print(f"kv_cache_capacity    : {row['kv_cache_capacity_mib']:.4f} MiB")
    print("-" * 100)

    print("Naive decode")
    print(
        f"per-step             : mean={row['naive_decode_step_mean_ms']:8.4f} ms   "
        f"std={row['naive_decode_step_std_ms']:8.4f}   "
        f"min={row['naive_decode_step_min_ms']:8.4f}   "
        f"max={row['naive_decode_step_max_ms']:8.4f}"
    )
    print(
        f"total                : mean={row['naive_decode_total_mean_ms']:8.4f} ms   "
        f"std={row['naive_decode_total_std_ms']:8.4f}   "
        f"min={row['naive_decode_total_min_ms']:8.4f}   "
        f"max={row['naive_decode_total_max_ms']:8.4f}"
    )
    print("-" * 100)

    print("KV-cache")
    print(
        f"prefill              : mean={row['cache_prefill_mean_ms']:8.4f} ms   "
        f"std={row['cache_prefill_std_ms']:8.4f}   "
        f"min={row['cache_prefill_min_ms']:8.4f}   "
        f"max={row['cache_prefill_max_ms']:8.4f}"
    )
    print(
        f"decode per-step      : mean={row['cache_decode_step_mean_ms']:8.4f} ms   "
        f"std={row['cache_decode_step_std_ms']:8.4f}   "
        f"min={row['cache_decode_step_min_ms']:8.4f}   "
        f"max={row['cache_decode_step_max_ms']:8.4f}"
    )
    print(
        f"decode total         : mean={row['cache_decode_total_mean_ms']:8.4f} ms   "
        f"std={row['cache_decode_total_std_ms']:8.4f}   "
        f"min={row['cache_decode_total_min_ms']:8.4f}   "
        f"max={row['cache_decode_total_max_ms']:8.4f}"
    )
    print("-" * 100)

    print("Speedup")
    print(f"step speedup         : {row['step_speedup']:.4f}x")
    print(f"total speedup        : {row['total_speedup']:.4f}x")

    if args.csv_out is not None:
        write_rows_to_csv(args.csv_out, [row])
        print("-" * 100)
        print(f"CSV written to       : {args.csv_out}")


if __name__ == "__main__":
    main()