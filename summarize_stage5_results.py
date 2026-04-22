# summarize_stage5_results.py

import argparse

import pandas as pd

from src.benchmark.stage4b_analysis import parse_bool_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 5 results.")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "all_correct_final_paths" in df.columns:
        df = df[parse_bool_series(df["all_correct_final_paths"])].copy()

    if len(df) == 0:
        raise ValueError("No valid rows found.")

    print("=" * 110)
    print("Stage 5 Summary")
    print("=" * 110)

    grouped = df.groupby(["resolved_device", "dtype_name"], dropna=False)

    for (device, dtype), group in grouped:
        group = group.sort_values("prompt_len")

        print(f"device={device}, dtype={dtype}")
        print(f"  rows                               : {len(group)}")
        print(f"  mean cache vs naive                : {group['cache_vs_naive_full_speedup'].mean():.4f}x")
        print(f"  mean Stage4A vs cache              : {group['stage4a_vs_cache_full_speedup'].mean():.4f}x")
        print(f"  mean compiled vs cache             : {group['compiled_vs_cache_full_speedup'].mean():.4f}x")

        graph_mask = parse_bool_series(group["backend_cuda_graph_available"])
        if graph_mask.any():
            graph_group = group[graph_mask]
            print(f"  mean CUDA Graph vs cache           : {graph_group['cuda_graph_vs_cache_full_speedup'].mean():.4f}x")
            print(f"  mean CUDA Graph vs compiled        : {graph_group['cuda_graph_vs_compiled_full_speedup'].mean():.4f}x")
        else:
            print("  CUDA Graph                         : not available in this group")

        print("  best final path counts             :")
        best_counts = group["best_final_path_name"].value_counts(dropna=False)
        for path_name, count in best_counts.items():
            print(f"    - {path_name}: {count}")

        print()

    print("=" * 110)


if __name__ == "__main__":
    main()
