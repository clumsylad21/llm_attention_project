import argparse

from src.benchmark.stage4_experiment import build_stage4_row, write_rows_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 sweep: naive vs KV-cache vs optimized decode."
    )

    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--dtypes", nargs="+", default=["fp32"])

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)

    parser.add_argument("--prompt-lens", nargs="+", type=int, required=True)
    parser.add_argument("--gen-steps-list", nargs="+", type=int, required=True)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--csv-out", type=str, required=True)

    args = parser.parse_args()

    rows = []

    configs = []
    for device in args.devices:
        for dtype in args.dtypes:
            for prompt_len in args.prompt_lens:
                for gen_steps in args.gen_steps_list:
                    configs.append((device, dtype, prompt_len, gen_steps))

    print("=" * 100)
    print("Stage 4 Sweep")
    print("=" * 100)
    print(f"num_configs            : {len(configs)}")
    print(f"devices                : {args.devices}")
    print(f"dtypes                 : {args.dtypes}")
    print(f"prompt_lens            : {args.prompt_lens}")
    print(f"gen_steps_list         : {args.gen_steps_list}")
    print("=" * 100)

    for idx, (device, dtype, prompt_len, gen_steps) in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] "
            f"device={device} dtype={dtype} prompt_len={prompt_len} gen_steps={gen_steps}"
        )

        try:
            row = build_stage4_row(
                device_requested=device,
                dtype_name=dtype,
                batch=args.batch,
                heads=args.heads,
                head_dim=args.head_dim,
                prompt_len=prompt_len,
                gen_steps=gen_steps,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
            rows.append(row)

        except Exception as exc:
            print(f"  FAILED: {exc}")

    if len(rows) == 0:
        raise RuntimeError("All sweep runs failed. No CSV written.")

    write_rows_to_csv(rows, args.csv_out)
    print()
    print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()