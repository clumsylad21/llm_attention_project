# run_stage5_sweep.py

import argparse

from src.benchmark.stage4b_experiment import write_rows_to_csv
from src.benchmark.stage5_experiment import build_stage5_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5 sweep: final curated comparison across the most important decode paths."
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

    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-cuda-graphs", action="store_true")

    parser.add_argument("--csv-out", type=str, required=True)

    args = parser.parse_args()

    rows = []
    configs = []

    for device in args.devices:
        for dtype in args.dtypes:
            for prompt_len in args.prompt_lens:
                for gen_steps in args.gen_steps_list:
                    configs.append((device, dtype, prompt_len, gen_steps))

    print("=" * 110)
    print("Stage 5 Sweep")
    print("=" * 110)
    print(f"num_configs                   : {len(configs)}")
    print(f"devices                       : {args.devices}")
    print(f"dtypes                        : {args.dtypes}")
    print(f"prompt_lens                   : {args.prompt_lens}")
    print(f"gen_steps_list                : {args.gen_steps_list}")
    print("=" * 110)

    for idx, (device, dtype, prompt_len, gen_steps) in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] "
            f"device={device} dtype={dtype} prompt_len={prompt_len} gen_steps={gen_steps}"
        )

        try:
            row = build_stage5_row(
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
                compile_mode=args.compile_mode,
                fullgraph=args.fullgraph,
                enable_compile=not args.disable_compile,
                enable_cuda_graphs=not args.disable_cuda_graphs,
            )
            rows.append(row)

            graph_text = (
                f"{row['cuda_graph_vs_cache_full_speedup']:.4f}x"
                if row["backend_cuda_graph_available"]
                else "n/a"
            )

            print(
                f"  all_correct={row['all_correct_final_paths']} "
                f"best_path={row['best_final_path_name']} "
                f"compiled_vs_cache={row['compiled_vs_cache_full_speedup']:.4f}x "
                f"graph_vs_cache={graph_text}"
            )

        except Exception as exc:
            print(f"  FAILED: {exc}")

    if len(rows) == 0:
        raise RuntimeError("All sweep runs failed. No CSV written.")

    write_rows_to_csv(rows, args.csv_out)
    print()
    print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
