# plot_stage6_results.py

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 6 latency curves from a CSV sweep.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="stage6_latency_plot.png")
    args = parser.parse_args()

    with open(args.csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    for row in rows:
        if row["resolved_device"] != "cuda":
            continue
        key = (row["dtype_name"], int(row["gen_steps"]))
        grouped[key].append(row)

    if len(grouped) == 0:
        raise ValueError("No CUDA rows found in CSV for plotting.")

    plt.figure(figsize=(9, 6))

    for (dtype, gen_steps), group_rows in sorted(grouped.items()):
        group_rows.sort(key=lambda r: int(r["prompt_len"]))
        x = [int(r["prompt_len"]) for r in group_rows]
        y_stage4a = [float(r["stage4a_full_total_mean_ms"]) for r in group_rows]
        y_compiled = [float(r["compiled_full_total_mean_ms"]) for r in group_rows]
        y_stage6 = [float(r["stage6_full_total_mean_ms"]) for r in group_rows]

        plt.plot(x, y_stage4a, marker="o", label=f"Stage4A ({dtype}, G={gen_steps})")
        plt.plot(x, y_compiled, marker="s", label=f"Compiled ({dtype}, G={gen_steps})")
        plt.plot(x, y_stage6, marker="^", label=f"Stage6 ({dtype}, G={gen_steps})")

    plt.xlabel("Prompt length")
    plt.ylabel("Full decode latency (ms)")
    plt.title("Stage 6 custom CUDA path vs existing decode paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
