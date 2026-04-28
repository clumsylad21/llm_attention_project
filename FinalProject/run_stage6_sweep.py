# run_stage6_sweep.py

import argparse

from src.benchmark.stage4b_experiment import write_rows_to_csv
from src.benchmark.stage6_experiment import build_stage6_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Stage 6 sweep across devices, dtypes, prompt lengths, and generation lengths."
    )
    parser.add_argument("--devices", nargs="+", default=["cuda"], choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtypes", nargs="+", default=["fp16", "fp32"], choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--gen-steps-list", nargs="+", type=int, default=[32])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-cuda-graphs", action="store_true")
    parser.add_argument("--disable-stage6", action="store_true")
    parser.add_argument("--csv-out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for device in args.devices:
        for dtype in args.dtypes:
            for prompt_len in args.prompt_lens:
                for gen_steps in args.gen_steps_list:
                    print(
                        f"running device={device} dtype={dtype} "
                        f"prompt_len={prompt_len} gen_steps={gen_steps}"
                    )
                    row = build_stage6_row(
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
                        enable_stage6=not args.disable_stage6,
                    )
                    rows.append(row)

    write_rows_to_csv(rows, args.csv_out)
    print(f"Wrote {len(rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
