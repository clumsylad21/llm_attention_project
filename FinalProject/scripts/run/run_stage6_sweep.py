# run_stage6_sweep.py

import argparse
import math
from typing import Any, Optional

from src.benchmark.stage6_experiment import build_stage6_row
from src.benchmark.sweep_utils import (
    add_backend_sweep_args,
    add_standard_sweep_args,
    run_sweep,
)


def stage6_row_summary(row: dict[str, Any]) -> Optional[str]:
    speedup = row["stage6_vs_cache_full_speedup"]

    speedup_text = f"{speedup:.4f}x"
    if isinstance(speedup, float) and math.isnan(speedup):
        speedup_text = "n/a"

    return (
        f"all_correct={row['all_correct_with_stage6']} "
        f"stage6_available={row['stage6_available']} "
        f"stage6_status={row['stage6_status']} "
        f"best_path={row['best_path_with_stage6_name']} "
        f"stage6_vs_cache={speedup_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Stage 6 sweep across devices, dtypes, prompt lengths, and generation lengths."
    )

    add_standard_sweep_args(
        parser,
        default_devices=["cuda"],
        default_dtypes=["fp16", "fp32"],
        default_prompt_lens=[128, 256, 512, 1024],
        default_gen_steps_list=[32],
        device_choices=["cpu", "cuda", "auto"],
        dtype_choices=["fp16", "bf16", "fp32"],
    )
    add_backend_sweep_args(parser)

    parser.add_argument("--disable-stage6", action="store_true")

    args = parser.parse_args()

    run_sweep(
        stage_name="Stage 6 Sweep",
        args=args,
        build_row=build_stage6_row,
        extra_build_kwargs={
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "enable_compile": not args.disable_compile,
            "enable_cuda_graphs": not args.disable_cuda_graphs,
            "enable_stage6": not args.disable_stage6,
        },
        row_summary=stage6_row_summary,
        width=110,
    )


if __name__ == "__main__":
    main()