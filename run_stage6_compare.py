# run_stage6_compare.py

import argparse
import math

from src.benchmark.stage4b_experiment import write_rows_to_csv
from src.benchmark.stage6_experiment import build_stage6_row


def fmt_or_na(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.4f}"

def fmt_speedup(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.4f}x"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: minimal custom CUDA attention prototype integrated into the existing decode pipeline."
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-cuda-graphs", action="store_true")
    parser.add_argument("--disable-stage6", action="store_true")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    row = build_stage6_row(
        device_requested=args.device,
        dtype_name=args.dtype,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        prompt_len=args.prompt_len,
        gen_steps=args.gen_steps,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        compile_mode=args.compile_mode,
        fullgraph=args.fullgraph,
        enable_compile=not args.disable_compile,
        enable_cuda_graphs=not args.disable_cuda_graphs,
        enable_stage6=not args.disable_stage6,
    )

    print("=" * 110)
    print("Stage 6 - Minimal Custom CUDA Attention Prototype")
    print("=" * 110)
    print(f"device requested            : {row['device_requested']}")
    print(f"resolved device             : {row['resolved_device']}")
    print(f"dtype                       : {row['dtype_name']} ({row['resolved_dtype']})")
    print(f"batch / heads / head_dim    : {row['batch']} / {row['heads']} / {row['head_dim']}")
    print(f"prompt_len / gen_steps      : {row['prompt_len']} / {row['gen_steps']}")
    print(f"Stage 5 best path           : {row['best_final_path_name']}")
    print(f"Stage 6 status              : {row['stage6_status']}")
    print(f"Stage 6 available           : {row['stage6_available']}")
    print(f"tile tokens                 : {row['stage6_tile_tokens']}")
    print("=" * 110)

    print("Correctness")
    print("-" * 110)
    print(
        f"Stage 6 vs naive    : allclose={row['stage6_vs_naive_allclose']} "
        f"max_abs_diff={row['stage6_vs_naive_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['stage6_vs_naive_mean_abs_diff']:.6e}"
    )
    print(
        f"Stage 6 vs Stage4A  : allclose={row['stage6_vs_stage4a_allclose']} "
        f"max_abs_diff={row['stage6_vs_stage4a_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['stage6_vs_stage4a_mean_abs_diff']:.6e}"
    )
    print(f"all correct with Stage 6    : {row['all_correct_with_stage6']}")

    print("=" * 110)
    print("Latency (full decode)")
    print("-" * 110)
    print(f"cache      : {row['cache_full_total_mean_ms']:.4f} ms")
    print(f"stage4a    : {row['stage4a_full_total_mean_ms']:.4f} ms")
    print(f"compiled   : {row['compiled_full_total_mean_ms']:.4f} ms")
    print(f"cuda graph : {fmt_or_na(row['cuda_graph_full_total_mean_ms'])} ms")
    print(f"stage6     : {fmt_or_na(row['stage6_full_total_mean_ms'])} ms")

    print("=" * 110)
    print("Speedups")
    print("-" * 110)
    print(f"stage6 vs cache      : {fmt_speedup(row['stage6_vs_cache_full_speedup'])}")
    print(f"stage6 vs stage4a    : {fmt_speedup(row['stage6_vs_stage4a_full_speedup'])}")
    print(f"stage6 vs compiled   : {fmt_speedup(row['stage6_vs_compiled_full_speedup'])}")
    print(f"stage6 vs cuda graph : {fmt_speedup(row['stage6_vs_cuda_graph_full_speedup'])}")
    print(f"stage6 vs best stage5: {fmt_speedup(row['stage6_vs_best_stage5_full_speedup'])}")
    print("=" * 110)
    print(f"best path incl. Stage 6     : {row['best_path_with_stage6_name']}")
    print(f"best latency incl. Stage 6  : {fmt_or_na(row['best_path_with_stage6_mean_ms'])} ms")
    print("=" * 110)

    if args.csv_out is not None:
        write_rows_to_csv([row], args.csv_out)
        print(f"Wrote CSV row to: {args.csv_out}")


if __name__ == "__main__":
    main()
