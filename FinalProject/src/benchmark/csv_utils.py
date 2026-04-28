import csv
import os
from typing import Any


def write_rows_to_csv(csv_path: str, rows: list[dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a CSV file.

    We build the union of all keys across all rows so this also works
    when some sweep rows contain error information and some contain
    full benchmark results.
    """
    if not rows:
        raise ValueError("rows is empty. Nothing to write.")

    parent_dir = os.path.dirname(csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)