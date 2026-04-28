# src/benchmark/stage5_analysis.py

import pandas as pd

from src.benchmark.stage4b_analysis import parse_bool_series


def load_and_filter(
    csv_path: str,
    device: str,
    dtype: str,
    batch: int,
    heads: int,
    head_dim: int,
    gen_steps: int,
) -> pd.DataFrame:
    """
    Load a Stage 5 CSV and filter to one device/dtype/config slice.
    """
    df = pd.read_csv(csv_path)

    correct_mask = parse_bool_series(df["all_correct_final_paths"])
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
