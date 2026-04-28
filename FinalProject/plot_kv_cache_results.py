import argparse
import csv
import os

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the main Stage 3 KV-cache results (clean 3-plot version)."
    )

    parser.add_argument(
        "--csv-in",
        type=str,
        required=True,
        help="Input CSV produced by run_kv_sweep.py",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots will be saved.",
    )

    # We keep these filters explicit so the plots stay clean and controlled.
    parser.add_argument("--device", type=str, required=True, help="Resolved device to plot, e.g. cpu or cuda")
    parser.add_argument("--dtype", type=str, required=True, help="dtype_name to plot, e.g. fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--gen-steps", type=int, required=True, help="Fixed generation length for prompt sweep")

    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_int(row: dict, key: str) -> int:
    return int(row[key])


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def load_valid_rows(csv_path: str) -> list[dict]:
    """
    Load only valid rows:
    - status must be ok
    - allclose must be True
    """
    rows = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue

            if "allclose" in row and not parse_bool(row["allclose"]):
                continue

            rows.append(row)

    return rows


def matches_target_config(row: dict, args: argparse.Namespace) -> bool:
    """
    Keep only rows matching one fixed configuration, while allowing prompt_len to vary.
    """
    if row["resolved_device"] != args.device:
        return False

    if row["dtype_name"] != args.dtype:
        return False

    if int(row["batch"]) != args.batch:
        return False

    if int(row["heads"]) != args.heads:
        return False

    if int(row["head_dim"]) != args.head_dim:
        return False

    if int(row["gen_steps"]) != args.gen_steps:
        return False

    return True


def save_plot(fig, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    rows = load_valid_rows(args.csv_in)
    rows = [row for row in rows if matches_target_config(row, args)]

    if len(rows) == 0:
        print("No valid rows found for the requested configuration.")
        return

    # Sort by prompt length because x-axis is prompt length
    rows.sort(key=lambda row: int(row["prompt_len"]))

    # Need at least 2 prompt lengths to make a scaling plot
    unique_prompt_lens = sorted({int(row["prompt_len"]) for row in rows})
    if len(unique_prompt_lens) < 2:
        print("Need at least 2 different prompt lengths to generate prompt-scaling plots.")
        return

    x = [to_int(row, "prompt_len") for row in rows]

    naive_step = [to_float(row, "naive_decode_step_mean_ms") for row in rows]
    cache_step = [to_float(row, "cache_decode_step_mean_ms") for row in rows]

    naive_total = [to_float(row, "naive_decode_total_mean_ms") for row in rows]
    cache_total = [to_float(row, "cache_decode_total_mean_ms") for row in rows]

    total_speedup = [to_float(row, "total_speedup") for row in rows]

    title_suffix = (
        f"{args.device} | {args.dtype} | "
        f"B={args.batch} H={args.heads} D={args.head_dim} | gen={args.gen_steps}"
    )

    file_suffix = (
        f"device-{args.device}_dtype-{args.dtype}_"
        f"b-{args.batch}_h-{args.heads}_d-{args.head_dim}_gen-{args.gen_steps}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------
    # Plot 1: Decode per-step latency vs prompt length
    # -------------------------------------------------
    fig = plt.figure()
    plt.plot(x, naive_step, marker="o", label="Naive decode per-step")
    plt.plot(x, cache_step, marker="o", label="KV-cache decode per-step")
    plt.xlabel("Prompt length")
    plt.ylabel("Latency (ms)")
    plt.title(f"Decode per-step latency vs prompt length | {title_suffix}")
    plt.grid(True)
    plt.legend()
    save_plot(
        fig,
        os.path.join(
            args.output_dir,
            f"step_latency_vs_prompt_{file_suffix}.png",
        ),
    )

    # -------------------------------------------------
    # Plot 2: Decode total latency vs prompt length
    # -------------------------------------------------
    fig = plt.figure()
    plt.plot(x, naive_total, marker="o", label="Naive decode total")
    plt.plot(x, cache_total, marker="o", label="KV-cache decode total")
    plt.xlabel("Prompt length")
    plt.ylabel("Latency (ms)")
    plt.title(f"Decode total latency vs prompt length | {title_suffix}")
    plt.grid(True)
    plt.legend()
    save_plot(
        fig,
        os.path.join(
            args.output_dir,
            f"total_latency_vs_prompt_{file_suffix}.png",
        ),
    )

    # -------------------------------------------------
    # Plot 3: Total speedup vs prompt length
    # -------------------------------------------------
    fig = plt.figure()
    plt.plot(x, total_speedup, marker="o", label="Naive / KV-cache")
    plt.xlabel("Prompt length")
    plt.ylabel("Speedup (x)")
    plt.title(f"Total speedup vs prompt length | {title_suffix}")
    plt.grid(True)
    plt.legend()
    save_plot(
        fig,
        os.path.join(
            args.output_dir,
            f"total_speedup_vs_prompt_{file_suffix}.png",
        ),
    )

    print(f"Wrote 3 plots to: {args.output_dir}")


if __name__ == "__main__":
    main()