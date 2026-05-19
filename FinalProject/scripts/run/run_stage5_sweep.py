# run_stage5_sweep.py

import argparse
from typing import Any, Optional

from src.benchmark.stage5_experiment import build_stage5_row
from src.benchmark.sweep_utils import (
    add_backend_sweep_args,
    add_standard_sweep_args,
    run_sweep,
)


def stage5_row_summary(row: dict[str, Any]) -> Optional[str]:
    graph_text = (
        f"{row['cuda_graph_vs_cache_full_speedup']:.4f}x"
        if row["backend_cuda_graph_available"]
        else "n/a"
    )

    return (
        f"all_correct={row['all_correct_final_paths']} "
        f"best_path={row['best_final_path_name']} "
        f"compiled_vs_cache={row['compiled_vs_cache_full_speedup']:.4f}x "
        f"graph_vs_cache={graph_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5 sweep: final curated comparison across the most important decode paths."
    )

    add_standard_sweep_args(
        parser,
        default_devices=["cpu"],
        default_dtypes=["fp32"],
    )
    add_backend_sweep_args(parser)

    args = parser.parse_args()

    run_sweep(
        stage_name="Stage 5 Sweep",
        args=args,
        build_row=build_stage5_row,
        extra_build_kwargs={
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "enable_compile": not args.disable_compile,
            "enable_cuda_graphs": not args.disable_cuda_graphs,
        },
        row_summary=stage5_row_summary,
        width=110,
    )


if __name__ == "__main__":
    main()