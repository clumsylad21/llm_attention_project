# summarize_stage4b_results.py

import argparse

import pandas as pd

from src.benchmark.stage4b_analysis import parse_bool_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 4B results.")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "all_correct_available_paths" in df.columns:
        df = df[parse_bool_series(df["all_correct_available_paths"])].copy()

    if len(df) == 0:
        raise ValueError("No valid rows found.")

    print("=" * 110)
    print("Stage 4B Summary")
    print("=" * 110)

    grouped = df.groupby(["resolved_device", "dtype_name"], dropna=False)

    for (device, dtype), group in grouped:
        group = group.sort_values("prompt_len")

        print(f"device={device}, dtype={dtype}")
        print(f"  rows                               : {len(group)}")
        print(f"  mean Stage4A vs cache              : {group['optimized_vs_cache_full_speedup'].mean():.4f}x")
        print(f"  mean 4B-A vs cache                 : {group['backend_eager_vs_cache_full_speedup'].mean():.4f}x")
        print(f"  mean 4B-B vs cache                 : {group['backend_compiled_vs_cache_full_speedup'].mean():.4f}x")
        print(f"  mean 4B-B vs 4B-A                  : {group['backend_compiled_vs_backend_eager_full_speedup'].mean():.4f}x")

        graph_mask = parse_bool_series(group["backend_cuda_graph_available"])
        if graph_mask.any():
            graph_group = group[graph_mask]
            print(f"  mean 4B-C vs cache                 : {graph_group['backend_cuda_graph_vs_cache_full_speedup'].mean():.4f}x")
            print(f"  mean 4B-C vs 4B-B                  : {graph_group['backend_cuda_graph_vs_backend_compiled_full_speedup'].mean():.4f}x")
        else:
            print("  4B-C CUDA Graph                    : not available in this group")

        print()

    print("=" * 110)


if __name__ == "__main__":
    main()