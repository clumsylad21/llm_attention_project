# plot_stage4b_results.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.benchmark.stage4b_analysis import load_and_filter, parse_bool_series


def plot_full_total_latency(df, out_path: str) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(df["prompt_len"], df["naive_full_total_mean_ms"], marker="o", label="Naive")
    plt.plot(df["prompt_len"], df["cache_full_total_mean_ms"], marker="o", label="KV-cache")
    plt.plot(df["prompt_len"], df["optimized_full_total_mean_ms"], marker="o", label="Stage4A")
    plt.plot(df["prompt_len"], df["backend_eager_full_total_mean_ms"], marker="o", label="4B-A Eager")
    plt.plot(df["prompt_len"], df["backend_compiled_full_total_mean_ms"], marker="o", label="4B-B Compile")

    if parse_bool_series(df["backend_cuda_graph_available"]).any():
        graph_df = df[parse_bool_series(df["backend_cuda_graph_available"])].copy()
        plt.plot(
            graph_df["prompt_len"],
            graph_df["backend_cuda_graph_full_total_mean_ms"],
            marker="o",
            label="4B-C CUDA Graph",
        )

    plt.xlabel("Prompt length")
    plt.ylabel("Full workflow latency (ms)")
    plt.title("Stage 4B: Full Workflow Latency vs Prompt Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cache_relative_speedups(df, out_path: str) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(df["prompt_len"], df["optimized_vs_cache_full_speedup"], marker="o", label="Stage4A vs KV-cache")
    plt.plot(df["prompt_len"], df["backend_eager_vs_cache_full_speedup"], marker="o", label="4B-A vs KV-cache")
    plt.plot(df["prompt_len"], df["backend_compiled_vs_cache_full_speedup"], marker="o", label="4B-B vs KV-cache")

    if parse_bool_series(df["backend_cuda_graph_available"]).any():
        graph_df = df[parse_bool_series(df["backend_cuda_graph_available"])].copy()
        plt.plot(
            graph_df["prompt_len"],
            graph_df["backend_cuda_graph_vs_cache_full_speedup"],
            marker="o",
            label="4B-C vs KV-cache",
        )

    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Prompt length")
    plt.ylabel("Speedup (x)")
    plt.title("Stage 4B: Relative to Existing KV-cache Path")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_backend_ladder(df, out_path: str) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["prompt_len"],
        df["backend_compiled_vs_backend_eager_full_speedup"],
        marker="o",
        label="4B-B vs 4B-A",
    )

    if parse_bool_series(df["backend_cuda_graph_available"]).any():
        graph_df = df[parse_bool_series(df["backend_cuda_graph_available"])].copy()
        plt.plot(
            graph_df["prompt_len"],
            graph_df["backend_cuda_graph_vs_backend_compiled_full_speedup"],
            marker="o",
            label="4B-C vs 4B-B",
        )

    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Prompt length")
    plt.ylabel("Speedup (x)")
    plt.title("Stage 4B: Backend Ladder Gains")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 4B comparison results.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--gen-steps", type=int, required=True)
    parser.add_argument("--out-prefix", type=str, default="stage4b")
    args = parser.parse_args()

    df = load_and_filter(
        csv_path=args.csv,
        device=args.device,
        dtype=args.dtype,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        gen_steps=args.gen_steps,
    )

    if len(df) == 0:
        raise ValueError("No matching rows found after filtering.")

    prefix = Path(args.out_prefix)

    plot_full_total_latency(df, f"{prefix}_full_total_latency.png")
    plot_cache_relative_speedups(df, f"{prefix}_cache_relative_speedups.png")
    plot_backend_ladder(df, f"{prefix}_backend_ladder.png")

    print(f"Wrote: {prefix}_full_total_latency.png")
    print(f"Wrote: {prefix}_cache_relative_speedups.png")
    print(f"Wrote: {prefix}_backend_ladder.png")


if __name__ == "__main__":
    main()