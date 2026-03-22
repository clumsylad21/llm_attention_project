# run_stage4b_compare.py

import argparse

from src.benchmark.stage4b_experiment import build_stage4b_row, write_rows_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4B: compare naive, KV-cache, Stage 4A, and backend ladder paths."
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

    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-cuda-graphs", action="store_true")

    parser.add_argument("--csv-out", type=str, default=None)

    args = parser.parse_args()

    row = build_stage4b_row(
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
        compile_mode=args.compile_mode,
        fullgraph=args.fullgraph,
        enable_compile=not args.disable_compile,
        enable_cuda_graphs=not args.disable_cuda_graphs,
    )

    print("=" * 110)
    print("Stage 4B - Backend Ladder Comparison")
    print("=" * 110)
    print(f"device requested              : {row['device_requested']}")
    print(f"resolved device               : {row['resolved_device']}")
    print(f"dtype                         : {row['dtype_name']} ({row['resolved_dtype']})")
    print(f"batch                         : {row['batch']}")
    print(f"heads                         : {row['heads']}")
    print(f"head_dim                      : {row['head_dim']}")
    print(f"model_dim                     : {row['model_dim']}")
    print(f"prompt_len                    : {row['prompt_len']}")
    print(f"gen_steps                     : {row['gen_steps']}")
    print(f"compile status                : {row['backend_compiled_status']}")
    print(f"cuda graph status             : {row['backend_cuda_graph_status']}")
    print("=" * 110)

    print("Correctness")
    print("-" * 110)
    print(
        f"naive vs cache                : allclose={row['naive_vs_cache_allclose']} "
        f"max_abs_diff={row['naive_vs_cache_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_cache_mean_abs_diff']:.6e}"
    )
    print(
        f"naive vs optimized           : allclose={row['naive_vs_optimized_allclose']} "
        f"max_abs_diff={row['naive_vs_optimized_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_optimized_mean_abs_diff']:.6e}"
    )
    print(
        f"naive vs backend eager       : allclose={row['naive_vs_backend_eager_allclose']} "
        f"max_abs_diff={row['naive_vs_backend_eager_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_backend_eager_mean_abs_diff']:.6e}"
    )
    print(
        f"naive vs backend compiled    : allclose={row['naive_vs_backend_compiled_allclose']} "
        f"max_abs_diff={row['naive_vs_backend_compiled_max_abs_diff']:.6e} "
        f"mean_abs_diff={row['naive_vs_backend_compiled_mean_abs_diff']:.6e}"
    )

    if row["backend_cuda_graph_available"]:
        print(
            f"naive vs backend cudagraph   : allclose={row['naive_vs_backend_cuda_graph_allclose']} "
            f"max_abs_diff={row['naive_vs_backend_cuda_graph_max_abs_diff']:.6e} "
            f"mean_abs_diff={row['naive_vs_backend_cuda_graph_mean_abs_diff']:.6e}"
        )
    else:
        print("naive vs backend cudagraph   : not available")

    print(f"all correct available paths  : {row['all_correct_available_paths']}")
    print()

    print("Memory")
    print("-" * 110)
    print(f"hidden states                : {row['hidden_states_mib']:.4f} MiB")
    print(f"separate weights             : {row['separate_weights_mib']:.4f} MiB")
    print(f"fused weights                : {row['fused_weights_mib']:.4f} MiB")
    print(f"standard KV capacity         : {row['standard_kv_cache_capacity_mib']:.4f} MiB")
    print(f"backend KV capacity          : {row['backend_kv_cache_capacity_mib']:.4f} MiB")
    print(f"backend output buffer        : {row['backend_output_buffer_mib']:.4f} MiB")
    print()

    print("Latency")
    print("-" * 110)
    print(
        f"naive full total             : {row['naive_full_total_mean_ms']:.4f} ms "
        f"(std={row['naive_full_total_std_ms']:.4f})"
    )
    print(
        f"cache full total             : {row['cache_full_total_mean_ms']:.4f} ms "
        f"(std={row['cache_full_total_std_ms']:.4f})"
    )
    print(
        f"optimized full total         : {row['optimized_full_total_mean_ms']:.4f} ms "
        f"(std={row['optimized_full_total_std_ms']:.4f})"
    )
    print(
        f"backend eager full total     : {row['backend_eager_full_total_mean_ms']:.4f} ms "
        f"(std={row['backend_eager_full_total_std_ms']:.4f})"
    )
    print(
        f"backend compiled full total  : {row['backend_compiled_full_total_mean_ms']:.4f} ms "
        f"(std={row['backend_compiled_full_total_std_ms']:.4f})"
    )

    if row["backend_cuda_graph_available"]:
        print(
            f"backend cudagraph total      : {row['backend_cuda_graph_full_total_mean_ms']:.4f} ms "
            f"(std={row['backend_cuda_graph_full_total_std_ms']:.4f})"
        )
    else:
        print("backend cudagraph total      : not available")

    print()

    print("Amortized per token")
    print("-" * 110)
    print(f"naive                         : {row['naive_amortized_per_step_mean_ms']:.4f} ms")
    print(f"cache                         : {row['cache_amortized_per_step_mean_ms']:.4f} ms")
    print(f"optimized                     : {row['optimized_amortized_per_step_mean_ms']:.4f} ms")
    print(f"backend eager                 : {row['backend_eager_amortized_per_step_mean_ms']:.4f} ms")
    print(f"backend compiled              : {row['backend_compiled_amortized_per_step_mean_ms']:.4f} ms")

    if row["backend_cuda_graph_available"]:
        print(f"backend cudagraph             : {row['backend_cuda_graph_amortized_per_step_mean_ms']:.4f} ms")
    else:
        print("backend cudagraph             : not available")

    print()
    print("Speedups")
    print("-" * 110)
    print(f"cache vs naive                : {row['cache_vs_naive_full_speedup']:.4f}x")
    print(f"optimized vs cache            : {row['optimized_vs_cache_full_speedup']:.4f}x")
    print(f"backend eager vs cache        : {row['backend_eager_vs_cache_full_speedup']:.4f}x")
    print(f"backend compiled vs cache     : {row['backend_compiled_vs_cache_full_speedup']:.4f}x")
    print(f"compiled vs eager             : {row['backend_compiled_vs_backend_eager_full_speedup']:.4f}x")

    if row["backend_cuda_graph_available"]:
        print(f"backend cudagraph vs cache    : {row['backend_cuda_graph_vs_cache_full_speedup']:.4f}x")
        print(f"cudagraph vs compiled         : {row['backend_cuda_graph_vs_backend_compiled_full_speedup']:.4f}x")
    else:
        print("backend cudagraph vs cache    : not available")
        print("cudagraph vs compiled         : not available")

    if args.csv_out is not None:
        write_rows_to_csv([row], args.csv_out)
        print()
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()