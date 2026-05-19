import argparse

from src.benchmark.stage4_experiment import build_stage4_row
from src.benchmark.sweep_utils import add_standard_sweep_args, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 sweep: naive vs KV-cache vs optimized decode."
    )

    add_standard_sweep_args(
        parser,
        default_devices=["cpu"],
        default_dtypes=["fp32"],
    )

    args = parser.parse_args()

    run_sweep(
        stage_name="Stage 4 Sweep",
        args=args,
        build_row=build_stage4_row,
        width=100,
    )


if __name__ == "__main__":
    main()