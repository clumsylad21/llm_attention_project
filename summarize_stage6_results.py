# summarize_stage6_results.py

import argparse
import csv
import math
from collections import defaultdict


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite_mean(values: list[float]) -> float:
    valid = [x for x in values if not math.isnan(x)]
    if len(valid) == 0:
        return float("nan")
    return sum(valid) / len(valid)


def fmt_or_na(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 6 CSV results.")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["resolved_device"], row["dtype_name"])].append(row)

    print("=" * 110)
    print("Stage 6 Summary")
    print("=" * 110)

    for (device, dtype), group_rows in sorted(grouped.items()):
        stage6_available_count = sum(
            row.get("stage6_available", "").lower() == "true" for row in group_rows
        )
        stage6_vs_stage4a = [_to_float(row, "stage6_vs_stage4a_full_speedup") for row in group_rows]
        stage6_vs_compiled = [_to_float(row, "stage6_vs_compiled_full_speedup") for row in group_rows]
        stage6_vs_cuda_graph = [_to_float(row, "stage6_vs_cuda_graph_full_speedup") for row in group_rows]

        best_counts: dict[str, int] = defaultdict(int)
        for row in group_rows:
            best_counts[row.get("best_path_with_stage6_name", "unknown")] += 1

        print(f"device={device}, dtype={dtype}")
        print(f"  rows                         : {len(group_rows)}")
        print(f"  stage6 available rows        : {stage6_available_count}")
        print(f"  mean stage6 vs stage4a       : {fmt_or_na(_finite_mean(stage6_vs_stage4a))}x")
        print(f"  mean stage6 vs compiled      : {fmt_or_na(_finite_mean(stage6_vs_compiled))}x")
        print(f"  mean stage6 vs cuda graph    : {fmt_or_na(_finite_mean(stage6_vs_cuda_graph))}x")
        print(f"  best path counts             : {dict(sorted(best_counts.items()))}")
        print()


if __name__ == "__main__":
    main()
