import argparse
import itertools

from src.benchmark.csv_utils import write_rows_to_csv
from src.benchmark.kv_experiment import (
    run_single_kv_experiment,
    build_error_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: sweep naive-vs-KV-cache experiments."
    )

    parser.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="List of prompt lengths to sweep.",
    )

    parser.add_argument(
        "--gen-steps-list",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="List of generation lengths to sweep.",
    )

    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["auto"],
        choices=["auto", "cpu", "cuda"],
        help="List of devices to sweep.",
    )

    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["fp32"],
        choices=["fp16", "bf16", "fp32"],
        help="List of dtypes to sweep.",
    )

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--csv-out",
        type=str,
        default="results/kv_cache_sweep.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configs = list(
        itertools.product(
            args.prompt_lens,
            args.gen_steps_list,
            args.devices,
            args.dtypes,
        )
    )

    rows = []

    print("=" * 100)
    print("Stage 3 - KV Cache Sweep")
    print("=" * 100)
    print(f"num_configs          : {len(configs)}")
    print("-" * 100)

    for idx, (prompt_len, gen_steps, device_str, dtype_str) in enumerate(configs, start=1):
        print(
            f"[{idx:>3}/{len(configs)}] "
            f"prompt_len={prompt_len:<5d} "
            f"gen_steps={gen_steps:<5d} "
            f"device={device_str:<5s} "
            f"dtype={dtype_str}"
        )

        try:
            row = run_single_kv_experiment(
                device_str=device_str,
                dtype_str=dtype_str,
                batch=args.batch,
                heads=args.heads,
                head_dim=args.head_dim,
                prompt_len=prompt_len,
                gen_steps=gen_steps,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )

            print(
                f"           allclose={row['allclose']}   "
                f"step_speedup={row['step_speedup']:.4f}x   "
                f"total_speedup={row['total_speedup']:.4f}x"
            )

        except Exception as e:
            row = build_error_row(
                device_str=device_str,
                dtype_str=dtype_str,
                batch=args.batch,
                heads=args.heads,
                head_dim=args.head_dim,
                prompt_len=prompt_len,
                gen_steps=gen_steps,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                error_message=str(e),
            )

            print(f"           ERROR: {e}")

        rows.append(row)

    write_rows_to_csv(args.csv_out, rows)

    print("-" * 100)
    print(f"CSV written to       : {args.csv_out}")


if __name__ == "__main__":
    main()