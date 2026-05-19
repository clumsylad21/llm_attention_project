# src/benchmark/sweep_utils.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from src.benchmark.experiment_utils import write_rows_to_csv


@dataclass(frozen=True)
class SweepPoint:
    """
    One point in a benchmark sweep.

    Example:
        device="cuda", dtype="fp16", prompt_len=512, gen_steps=32
    """

    device: str
    dtype: str
    prompt_len: int
    gen_steps: int


def add_standard_sweep_args(
    parser: argparse.ArgumentParser,
    *,
    default_devices: Sequence[str],
    default_dtypes: Sequence[str],
    default_prompt_lens: Optional[Sequence[int]] = None,
    default_gen_steps_list: Optional[Sequence[int]] = None,
    device_choices: Optional[Sequence[str]] = None,
    dtype_choices: Optional[Sequence[str]] = None,
) -> None:
    """
    Add the common sweep arguments used by the staged benchmark runners.

    If prompt/gen defaults are not provided, those arguments are required.
    This preserves Stage 4/4B/5 behavior while still allowing Stage 6-style
    defaults later.
    """

    parser.add_argument(
        "--devices",
        nargs="+",
        default=list(default_devices),
        choices=list(device_choices) if device_choices is not None else None,
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=list(default_dtypes),
        choices=list(dtype_choices) if dtype_choices is not None else None,
    )

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)

    prompt_lens_kwargs: dict[str, Any] = {
        "nargs": "+",
        "type": int,
    }
    if default_prompt_lens is None:
        prompt_lens_kwargs["required"] = True
    else:
        prompt_lens_kwargs["default"] = list(default_prompt_lens)

    parser.add_argument("--prompt-lens", **prompt_lens_kwargs)

    gen_steps_kwargs: dict[str, Any] = {
        "nargs": "+",
        "type": int,
    }
    if default_gen_steps_list is None:
        gen_steps_kwargs["required"] = True
    else:
        gen_steps_kwargs["default"] = list(default_gen_steps_list)

    parser.add_argument("--gen-steps-list", **gen_steps_kwargs)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--csv-out", type=str, required=True)


def add_backend_sweep_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common backend optimization flags used by Stage 4B/5/6 runners.

    Stage 4 does not need these, but keeping this helper here lets us refactor
    Stage 4B/5/6 one by one later without repeating these flags.
    """

    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-cuda-graphs", action="store_true")


def build_sweep_points(args: argparse.Namespace) -> list[SweepPoint]:
    """
    Expand command-line sweep lists into concrete sweep points.
    """

    points: list[SweepPoint] = []

    for device in args.devices:
        for dtype in args.dtypes:
            for prompt_len in args.prompt_lens:
                for gen_steps in args.gen_steps_list:
                    points.append(
                        SweepPoint(
                            device=device,
                            dtype=dtype,
                            prompt_len=prompt_len,
                            gen_steps=gen_steps,
                        )
                    )

    return points


def build_common_row_kwargs(
    args: argparse.Namespace,
    point: SweepPoint,
) -> dict[str, Any]:
    """
    Build the common keyword arguments expected by stage row builders.
    """

    return {
        "device_requested": point.device,
        "dtype_name": point.dtype,
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "prompt_len": point.prompt_len,
        "gen_steps": point.gen_steps,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
    }


def print_sweep_header(
    *,
    stage_name: str,
    args: argparse.Namespace,
    num_points: int,
    width: int,
) -> None:
    """
    Print a consistent sweep header.
    """

    print("=" * width)
    print(stage_name)
    print("=" * width)
    print(f"num_configs            : {num_points}")
    print(f"devices                : {args.devices}")
    print(f"dtypes                 : {args.dtypes}")
    print(f"prompt_lens            : {args.prompt_lens}")
    print(f"gen_steps_list         : {args.gen_steps_list}")
    print("=" * width)


def print_sweep_point(
    *,
    index: int,
    total: int,
    point: SweepPoint,
) -> None:
    """
    Print one sweep point before running it.
    """

    print(
        f"[{index}/{total}] "
        f"device={point.device} "
        f"dtype={point.dtype} "
        f"prompt_len={point.prompt_len} "
        f"gen_steps={point.gen_steps}"
    )


def run_sweep(
    *,
    stage_name: str,
    args: argparse.Namespace,
    build_row: Callable[..., dict[str, Any]],
    extra_build_kwargs: Optional[dict[str, Any]] = None,
    row_summary: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
    width: int = 110,
) -> list[dict[str, Any]]:
    """
    Run a benchmark sweep and write the resulting rows to CSV.

    This intentionally keeps the abstraction small:
    - common args become SweepPoint objects
    - common build-row kwargs are created here
    - stage-specific flags can be passed through extra_build_kwargs
    - stage-specific progress text can be passed through row_summary
    """

    points = build_sweep_points(args)

    print_sweep_header(
        stage_name=stage_name,
        args=args,
        num_points=len(points),
        width=width,
    )

    rows: list[dict[str, Any]] = []

    for idx, point in enumerate(points, start=1):
        print_sweep_point(index=idx, total=len(points), point=point)

        try:
            kwargs = build_common_row_kwargs(args, point)
            if extra_build_kwargs is not None:
                kwargs.update(extra_build_kwargs)

            row = build_row(**kwargs)
            rows.append(row)

            if row_summary is not None:
                summary = row_summary(row)
                if summary:
                    print(f"  {summary}")

        except Exception as exc:
            print(f"  FAILED: {exc}")

    if len(rows) == 0:
        raise RuntimeError("All sweep runs failed. No CSV written.")

    write_rows_to_csv(rows, args.csv_out)
    print()
    print(f"Wrote CSV: {args.csv_out}")

    return rows
