# run_stage5_compare.py

import argparse
import math

from src.benchmark.stage4b_experiment import write_rows_to_csv
from src.benchmark.stage5_experiment import build_stage5_row


def fmt_or_na(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: final curated comparison across the most important decode paths."
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
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

    parser.add_argument("--csv-out", type=str, default=None)

    args = parser.parse_args()

    row = build_stage5_row(
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
    )

    print("=" * 110)
    print("Stage 5 - Final Curated Comparison")
    print("=" * 110)
    print(f"device requested              : {row['device_requested']}")
    print(f"resolved device               : {row['resolved_device']}")
    print(f"dtype                         : {row['dtype_name']} ({row['resolved_dtype']})")
    print(f"batch                         : {row['batch']}")
    print(f"heads                         : {row['heads']}")
    print(f"head_dim                      : {row['head_dim']}")
    print(f"model_dim                     : {row['model_dim']}")
    print(f"prompt_len                    : {row['prompt_len']}")
    print(f"gen_steps                     : {row['gen_steps']}")
    print(f"final path keys               : {row['stage5_final_path_keys']}")
    print(f"excluded path keys            : {row['stage5_excluded_path_keys']}")
    print(f"compile status                : {row['backend_compiled_status']}")
    print(f"cuda graph status             : {row['backend_cuda_graph_status']}")
    print("=" * 110)

    print("Correctness")
    print("-" * 110)
    print(
        f"naive vs cache                : allclose={row['naive_vs_cache_allclose']} "
        f"max_abs_diff={row['naive_vs_cache_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_cache_mean_abs_diff']:.6e}"
    )
    print(
        f"naive vs stage4a              : allclose={row['naive_vs_stage4a_allclose']} "
        f"max_abs_diff={row['naive_vs_stage4a_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_stage4a_mean_abs_diff']:.6e}"
    )
    print(
        f"naive vs compiled             : allclose={row['naive_vs_compiled_allclose']} "
        f"max_abs_diff={row['naive_vs_compiled_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_compiled_mean_abs_diff']:.6e}"
    )

    if row["backend_cuda_graph_available"]:
        print(
            f"naive vs cuda graph           : allclose={row['naive_vs_cuda_graph_allclose']} "
            f"max_abs_diff={row['naive_vs_cuda_graph_max_abs_diff']:.6e} "
            f"mean_abs_diff={row['naive_vs_cuda_graph_mean_abs_diff']:.6e}"
        )
    else:
        print("naive vs cuda graph           : not available")

    print(f"all correct final paths       : {row['all_correct_final_paths']}")
    print()

    print("Latency")
    print("-" * 110)
    print(
        f"naive full total              : {row['naive_full_total_mean_ms']:.4f} ms "
        f"(std={row['naive_full_total_std_ms']:.4f})"
    )
    print(
        f"cache full total              : {row['cache_full_total_mean_ms']:.4f} ms "
        f"(std={row['cache_full_total_std_ms']:.4f})"
    )
    print(
        f"stage4a full total            : {row['stage4a_full_total_mean_ms']:.4f} ms "
        f"(std={row['stage4a_full_total_std_ms']:.4f})"
    )
    print(
        f"compiled full total           : {row['compiled_full_total_mean_ms']:.4f} ms "
        f"(std={row['compiled_full_total_std_ms']:.4f})"
    )

    if row["backend_cuda_graph_available"]:
        print(
            f"cuda graph full total         : {row['cuda_graph_full_total_mean_ms']:.4f} ms "
            f"(std={row['cuda_graph_full_total_std_ms']:.4f})"
        )
    else:
        print("cuda graph full total         : not available")

    print()
    print("Amortized per token")
    print("-" * 110)
    print(f"naive                         : {row['naive_amortized_per_step_mean_ms']:.4f} ms")
    print(f"cache                         : {row['cache_amortized_per_step_mean_ms']:.4f} ms")
    print(f"stage4a                       : {row['stage4a_amortized_per_step_mean_ms']:.4f} ms")
    print(f"compiled                      : {row['compiled_amortized_per_step_mean_ms']:.4f} ms")

    if row["backend_cuda_graph_available"]:
        print(f"cuda graph                    : {row['cuda_graph_amortized_per_step_mean_ms']:.4f} ms")
    else:
        print("cuda graph                    : not available")

    print()
    print("Speedups")
    print("-" * 110)
    print(f"cache vs naive                : {fmt_or_na(row['cache_vs_naive_full_speedup'])}x")
    print(f"stage4a vs naive              : {fmt_or_na(row['stage4a_vs_naive_full_speedup'])}x")
    print(f"stage4a vs cache              : {fmt_or_na(row['stage4a_vs_cache_full_speedup'])}x")
    print(f"compiled vs naive             : {fmt_or_na(row['compiled_vs_naive_full_speedup'])}x")
    print(f"compiled vs cache             : {fmt_or_na(row['compiled_vs_cache_full_speedup'])}x")

    if row["backend_cuda_graph_available"]:
        print(f"cuda graph vs naive           : {fmt_or_na(row['cuda_graph_vs_naive_full_speedup'])}x")
        print(f"cuda graph vs cache           : {fmt_or_na(row['cuda_graph_vs_cache_full_speedup'])}x")
        print(f"cuda graph vs compiled        : {fmt_or_na(row['cuda_graph_vs_compiled_full_speedup'])}x")
    else:
        print("cuda graph vs naive           : not available")
        print("cuda graph vs cache           : not available")
        print("cuda graph vs compiled        : not available")

    print()
    print("Best final path")
    print("-" * 110)
    print(f"name                          : {row['best_final_path_name']}")
    print(f"mean latency                  : {fmt_or_na(row['best_final_path_mean_ms'])} ms")

    if args.csv_out is not None:
        write_rows_to_csv([row], args.csv_out)
        print()
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
