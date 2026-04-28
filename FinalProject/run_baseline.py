import argparse
import csv
from pathlib import Path

from src.attention.sdpa_baseline import (
    make_decode_tensors,
    make_prefill_tensors,
    run_decode_step_sdpa,
    run_prefill_sdpa,
    tensor_mib,
)
from src.benchmark.timer import benchmark_function
from src.common.device import choose_device, choose_dtype, get_device_name
from src.common.seed import set_seed


def print_header(device: str, dtype, batch: int, heads: int, head_dim: int, warmup: int, iters: int):
    print("=" * 95)
    print("PyTorch SDPA Baseline")
    print("=" * 95)
    print(f"device      : {device}")
    print(f"hardware    : {get_device_name(device)}")
    print(f"dtype       : {dtype}")
    print(f"batch       : {batch}")
    print(f"heads       : {heads}")
    print(f"head_dim    : {head_dim}")
    print(f"warmup      : {warmup}")
    print(f"iters       : {iters}")
    print("=" * 95)
    print()


def print_table_row(mode, seq_len, result, q_mib, k_mib, v_mib, out_mib):
    total_in_mib = q_mib + k_mib + v_mib
    print(
        f"{mode:<14} "
        f"S={seq_len:<6} "
        f"mean={result.mean_ms:>8.3f} ms   "
        f"std={result.std_ms:>7.3f}   "
        f"min={result.min_ms:>7.3f}   "
        f"max={result.max_ms:>7.3f}   "
        f"in={total_in_mib:>8.2f} MiB   "
        f"out={out_mib:>7.2f} MiB   "
        f"chk={result.checksum:>10.6f}"
    )


def maybe_write_csv(csv_path: str, rows: list) -> None:
    if csv_path is None:
        return

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "seq_len",
        "batch",
        "heads",
        "head_dim",
        "device",
        "dtype",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "input_mib",
        "output_mib",
        "checksum",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"[saved] CSV written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 PyTorch SDPA baseline")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    set_seed(args.seed)

    print_header(
        device=device,
        dtype=dtype,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        iters=args.iters,
    )

    csv_rows = []

    for seq_len in args.seq_lens:
        # -------------------------
        # Prefill benchmark
        # -------------------------
        q, k, v = make_prefill_tensors(
            batch=args.batch,
            heads=args.heads,
            seq_len=seq_len,
            head_dim=args.head_dim,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

        prefill_result = benchmark_function(
            lambda: run_prefill_sdpa(q, k, v),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )

        prefill_out = run_prefill_sdpa(q, k, v)
        q_mib = tensor_mib(q)
        k_mib = tensor_mib(k)
        v_mib = tensor_mib(v)
        out_mib = tensor_mib(prefill_out)

        print_table_row(
            mode="prefill",
            seq_len=seq_len,
            result=prefill_result,
            q_mib=q_mib,
            k_mib=k_mib,
            v_mib=v_mib,
            out_mib=out_mib,
        )

        csv_rows.append(
            {
                "mode": "prefill",
                "seq_len": seq_len,
                "batch": args.batch,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "mean_ms": prefill_result.mean_ms,
                "std_ms": prefill_result.std_ms,
                "min_ms": prefill_result.min_ms,
                "max_ms": prefill_result.max_ms,
                "input_mib": q_mib + k_mib + v_mib,
                "output_mib": out_mib,
                "checksum": prefill_result.checksum,
            }
        )

        # -------------------------
        # Single-token decode benchmark
        # -------------------------
        q_dec, k_dec, v_dec = make_decode_tensors(
            batch=args.batch,
            heads=args.heads,
            context_len=seq_len,
            head_dim=args.head_dim,
            device=device,
            dtype=dtype,
            seed=args.seed + 1,
        )

        decode_result = benchmark_function(
            lambda: run_decode_step_sdpa(q_dec, k_dec, v_dec),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )

        decode_out = run_decode_step_sdpa(q_dec, k_dec, v_dec)
        q_dec_mib = tensor_mib(q_dec)
        k_dec_mib = tensor_mib(k_dec)
        v_dec_mib = tensor_mib(v_dec)
        out_dec_mib = tensor_mib(decode_out)

        print_table_row(
            mode="decode_step",
            seq_len=seq_len,
            result=decode_result,
            q_mib=q_dec_mib,
            k_mib=k_dec_mib,
            v_mib=v_dec_mib,
            out_mib=out_dec_mib,
        )

        csv_rows.append(
            {
                "mode": "decode_step",
                "seq_len": seq_len,
                "batch": args.batch,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "mean_ms": decode_result.mean_ms,
                "std_ms": decode_result.std_ms,
                "min_ms": decode_result.min_ms,
                "max_ms": decode_result.max_ms,
                "input_mib": q_dec_mib + k_dec_mib + v_dec_mib,
                "output_mib": out_dec_mib,
                "checksum": decode_result.checksum,
            }
        )

        print("-" * 95)

    maybe_write_csv(args.csv, csv_rows)


if __name__ == "__main__":
    main()