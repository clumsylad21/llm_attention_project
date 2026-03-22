import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_bool_series(series: pd.Series) -> pd.Series:
    """
    CSV booleans sometimes come back as strings.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_and_filter(
    csv_path: str,
    device: str,
    dtype: str,
    batch: int,
    heads: int,
    head_dim: int,
    gen_steps: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only rows where all method outputs matched closely.
    correct_mask = (
        parse_bool_series(df["naive_vs_cache_allclose"])
        & parse_bool_series(df["naive_vs_optimized_allclose"])
        & parse_bool_series(df["cache_vs_optimized_allclose"])
    )

    df = df[correct_mask]

    df = df[
        (df["resolved_device"] == device)
        & (df["dtype_name"] == dtype)
        & (df["batch"] == batch)
        & (df["heads"] == heads)
        & (df["head_dim"] == head_dim)
        & (df["gen_steps"] == gen_steps)
    ].copy()

    df = df.sort_values("prompt_len")
    return df


def plot_full_total_latency(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["prompt_len"], df["naive_full_total_mean_ms"], marker="o", label="Naive")
    plt.plot(df["prompt_len"], df["cache_full_total_mean_ms"], marker="o", label="KV-cache")
    plt.plot(df["prompt_len"], df["optimized_full_total_mean_ms"], marker="o", label="Optimized")

    plt.xlabel("Prompt length")
    plt.ylabel("Full workflow latency (ms)")
    plt.title("Stage 4: Full Workflow Latency vs Prompt Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_amortized_per_token(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["prompt_len"], df["naive_amortized_per_step_mean_ms"], marker="o", label="Naive")
    plt.plot(df["prompt_len"], df["cache_amortized_per_step_mean_ms"], marker="o", label="KV-cache")
    plt.plot(df["prompt_len"], df["optimized_amortized_per_step_mean_ms"], marker="o", label="Optimized")

    plt.xlabel("Prompt length")
    plt.ylabel("Amortized latency per generated token (ms)")
    plt.title("Stage 4: Amortized Per-Token Latency vs Prompt Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_speedups(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["prompt_len"], df["cache_vs_naive_full_speedup"], marker="o", label="KV-cache vs Naive")
    plt.plot(df["prompt_len"], df["optimized_vs_cache_full_speedup"], marker="o", label="Optimized vs KV-cache")
    plt.plot(df["prompt_len"], df["optimized_vs_naive_full_speedup"], marker="o", label="Optimized vs Naive")

    plt.xlabel("Prompt length")
    plt.ylabel("Speedup (x)")
    plt.title("Stage 4: Full Workflow Speedups vs Prompt Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 4 comparison results.")
    parser.add_argument("--csv", type=str, required=True)

    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--dtype", type=str, required=True)

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--gen-steps", type=int, required=True)

    parser.add_argument("--out-prefix", type=str, default="stage4")

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
    plot_amortized_per_token(df, f"{prefix}_amortized_per_token.png")
    plot_speedups(df, f"{prefix}_speedups.png")

    print(f"Wrote: {prefix}_full_total_latency.png")
    print(f"Wrote: {prefix}_amortized_per_token.png")
    print(f"Wrote: {prefix}_speedups.png")


if __name__ == "__main__":
    main()