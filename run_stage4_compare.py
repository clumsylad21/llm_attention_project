import argparse

from src.benchmark.stage4_experiment import build_stage4_row, write_rows_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: compare naive, KV-cache, and optimized decode paths."
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)

    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-steps", type=int, default=32)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--csv-out", type=str, default=None)

    args = parser.parse_args()

    row = build_stage4_row(
        device_requested=args.device,
        dtype_name=args.dtype,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        prompt_len=args.prompt_len,
        gen_steps=args.gen_steps,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )

    print("=" * 100)
    print("Stage 4 - Optimized Decode Comparison")
    print("=" * 100)
    print(f"device requested       : {row['device_requested']}")
    print(f"resolved device        : {row['resolved_device']}")
    print(f"dtype                  : {row['dtype_name']} ({row['resolved_dtype']})")
    print(f"batch                  : {row['batch']}")
    print(f"heads                  : {row['heads']}")
    print(f"head_dim               : {row['head_dim']}")
    print(f"model_dim              : {row['model_dim']}")
    print(f"prompt_len             : {row['prompt_len']}")
    print(f"gen_steps              : {row['gen_steps']}")
    print(f"optimized path         : {row['optimized_path_name']}")
    print("=" * 100)

    print("Correctness")
    print("-" * 100)
    print(
        f"naive vs cache         : allclose={row['naive_vs_cache_allclose']}   "
        f"max_abs_diff={row['naive_vs_cache_max_abs_diff']:.6e}   "
        f"mean_abs_diff={row['naive_vs_cache_mean_abs_diff']:.6e}"
    )
    print(
        f"naive vs optimized     : allclose={row['naive_vs_optimized_allclose']}   "
        f"max_abs_diff={row['naive_vs_optimized_max_abs_diff']:.6e}   "
        f"mean_abs_diff={row['naive_vs_optimized_mean_abs_diff']:.6e}"
    )
    print(
        f"cache vs optimized     : allclose={row['cache_vs_optimized_allclose']}   "
        f"max_abs_diff={row['cache_vs_optimized_max_abs_diff']:.6e}   "
        f"mean_abs_diff={row['cache_vs_optimized_mean_abs_diff']:.6e}"
    )
    print()

    print("Memory")
    print("-" * 100)
    print(f"hidden states          : {row['hidden_states_mib']:.4f} MiB")
    print(f"separate weights       : {row['separate_weights_mib']:.4f} MiB")
    print(f"fused weights          : {row['fused_weights_mib']:.4f} MiB")
    print(f"standard KV capacity   : {row['standard_kv_cache_capacity_mib']:.4f} MiB")
    print(f"optimized KV capacity  : {row['optimized_kv_cache_capacity_mib']:.4f} MiB")
    print()

    print("Latency")
    print("-" * 100)
    print(
        f"naive full total       : {row['naive_full_total_mean_ms']:.4f} ms   "
        f"(std={row['naive_full_total_std_ms']:.4f})"
    )
    print(
        f"cache full total       : {row['cache_full_total_mean_ms']:.4f} ms   "
        f"(std={row['cache_full_total_std_ms']:.4f})"
    )
    print(
        f"optimized full total   : {row['optimized_full_total_mean_ms']:.4f} ms   "
        f"(std={row['optimized_full_total_std_ms']:.4f})"
    )
    print(
        f"optimized prefill only : {row['optimized_prefill_only_mean_ms']:.4f} ms   "
        f"(fraction={row['optimized_prefill_fraction']:.4f})"
    )
    print(
        f"naive amortized/token  : {row['naive_amortized_per_step_mean_ms']:.4f} ms"
    )
    print(
        f"cache amortized/token  : {row['cache_amortized_per_step_mean_ms']:.4f} ms"
    )
    print(
        f"opt amortized/token    : {row['optimized_amortized_per_step_mean_ms']:.4f} ms"
    )
    print()

    print("Speedups")
    print("-" * 100)
    print(f"cache vs naive         : {row['cache_vs_naive_full_speedup']:.4f}x")
    print(f"optimized vs naive     : {row['optimized_vs_naive_full_speedup']:.4f}x")
    print(f"optimized vs cache     : {row['optimized_vs_cache_full_speedup']:.4f}x")

    if args.csv_out is not None:
        write_rows_to_csv([row], args.csv_out)
        print()
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()