# LLM Inference Optimization: Causal Attention, KV Caching, and CUDA Decode Paths

> A staged performance study of decoder-style LLM attention inference using KV caching, PyTorch execution-path improvements, CUDA Graph replay, and a minimal custom CUDA attention kernel.

![Python](https://img.shields.io/badge/language-Python-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![CUDA](https://img.shields.io/badge/accelerated_with-CUDA-green)
![Status](https://img.shields.io/badge/status-implemented-success)

## Overview

This repository explores practical performance optimization techniques for decoder-style LLM attention during autoregressive generation.

The project focuses on improving decode-time efficiency by comparing naive attention recomputation, KV-cache-based decoding, PyTorch-level optimized decode paths, backend/runtime optimized paths, and a minimal custom CUDA attention kernel.

Rather than treating inference as a single black box, the repository isolates and evaluates the major sources of latency in decoder-style attention. The project separates:

- algorithmic savings from KV caching,
- PyTorch-visible restructuring,
- backend/runtime savings from `torch.compile` and CUDA Graph replay,
- custom-kernel experimentation through a small CUDA extension.

The goal is not to build a production FlashAttention replacement. The goal is to build a clean, educational, measurable project that shows where decode-time performance improvements actually come from.

## Motivation

In decoder-only large language models, token generation happens one step at a time. Even when each decode step is small, the repeated sequence of attention, cache access, tensor operations, and GPU launches can dominate end-to-end generation time.

KV caching removes redundant recomputation over the prefix, but GPU decode latency can still remain high because the workload becomes many small repeated operations. This makes decoder-style attention a useful systems/HPC case study.

This project studies that gap directly by answering questions such as:

- How much does KV caching help compared to naive recomputation?
- What parts of decode latency remain after algorithmic optimization?
- How much additional speedup can be obtained from compiler and runtime techniques?
- Can a small custom CUDA kernel improve the inner single-query attention core?
- Why does a custom kernel not automatically beat well-integrated runtime paths?

## Project Stages

The project is organized as a staged performance study:

| Stage | Focus | Purpose |
|---|---|---|
| Stage 1 | PyTorch SDPA baseline | Establish baseline prefill/decode attention timing. |
| Stage 2 | Naive decode vs. KV-cache decode | Show the main algorithmic gain from avoiding repeated prefix recomputation. |
| Stage 3 | Benchmarking framework | Add sweeps, CSV logging, plotting, and reproducible experiment scripts. |
| Stage 4A | PyTorch-level optimized decode | Test fused QKV projection, cache-aware layouts, and manual single-query attention. |
| Stage 4B | Backend/runtime optimization | Compare eager execution, `torch.compile`, and CUDA Graph replay. |
| Stage 5 | Curated final comparison | Compare the main execution paths in one final benchmark workflow. |
| Stage 6 | Minimal custom CUDA kernel | Implement an educational CUDA prototype for the single-query decode attention core. |

## Implemented Work

This repository includes:

- a modular PyTorch prototype for decoder-style causal attention
- prefill vs. decode benchmarking
- naive recomputation vs. KV-cache comparison
- reusable CPU/GPU benchmarking pipelines
- correctness checks across execution paths
- automated sweeps with CSV logging and plotting
- Stage 4A optimized decode using fused QKV projection and decode-friendly KV-cache layouts
- Stage 4B backend comparisons across eager execution, preallocated execution, `torch.compile`, and CUDA Graph replay
- Stage 5 curated final comparison across naive decode, KV-cache decode, Stage 4A, compiled decode, and CUDA Graph replay
- Stage 6 minimal custom CUDA single-query attention kernel integrated into the PyTorch benchmark framework
- Stage 6 CUDA-side improvements including shared-memory tiling, adaptive block sizing, online softmax accumulation, configurable tile size, and output-buffer reuse

## Results Snapshot

The main result is that decoder-style attention performance improves in layers.

Key observations:

- Naive decode is slow because it repeatedly recomputes attention over the prefix.
- KV caching gives the first major algorithmic improvement by reusing previously computed keys and values.
- PyTorch-level restructuring in Stage 4A is educational but provides limited additional speedup beyond KV caching.
- Backend/runtime paths in Stage 4B are much stronger on GPU.
- `torch.compile` improves the decode execution path.
- CUDA Graph replay gives the strongest end-to-end GPU performance by reducing repeated launch overhead.
- Stage 6 successfully builds, runs, integrates with the benchmark framework, and passes correctness checks.
- The trusted Stage 6 tile-32 experiments show that the custom CUDA kernel is technically meaningful, but it still does not outperform compiled decode or CUDA Graph replay.

Representative trusted Stage 6 comparison:

| Path | FP16 latency (ms) | FP32 latency (ms) |
|---|---:|---:|
| KV-cache decode | 5.1475 | 4.6870 |
| Stage 4A optimized decode | 5.0241 | 5.9503 |
| Compiled backend decode | 1.7022 | 1.6778 |
| CUDA Graph decode | 0.7158 | 0.5741 |
| Stage 6 custom CUDA decode | 7.1602 | 6.9803 |

Trusted Stage 6 sweep summary using tile size 32:

| dtype | Rows | Stage6 / Stage4A | Stage6 / Compiled | Stage6 / CUDA Graph | Best path |
|---|---:|---:|---:|---:|---|
| FP16 | 4 | 0.8956x | 0.3399x | 0.1313x | CUDA Graph |
| FP32 | 4 | 0.7993x | 0.3470x | 0.1012x | CUDA Graph |

Overall insight:

**KV caching provides the main algorithmic gain, while compiled execution and CUDA Graph replay provide the strongest backend/runtime gains. A small custom CUDA kernel is useful and educational, but kernel-local optimization alone does not automatically beat well-integrated runtime paths.**

To become competitive, a custom CUDA path would likely need broader fusion, shape-specialized kernel tuning, improved memory layout, and integration with low-overhead execution mechanisms such as CUDA Graph replay.

## Repository Layout

```text
run_baseline.py                         # Stage 1 PyTorch SDPA baseline
run_kv_cache.py                         # Stage 2 naive decode vs KV-cache decode
run_kv_sweep.py                         # Stage 3 KV-cache sweeps and CSV generation
plot_kv_cache_results.py                # KV-cache result plotting
plot_kv_gpu_results.py                  # GPU KV-cache plotting

run_stage4_compare.py                   # Stage 4A optimized decode comparison
run_stage4_sweep.py                     # Stage 4A sweep experiments
plot_stage4_results.py                  # Stage 4A plotting utilities
summarize_stage4_results.py             # Stage 4A result summaries

run_stage4b_compare.py                  # Stage 4B eager vs compile vs CUDA Graph comparison
run_stage4b_sweep.py                    # Stage 4B backend-oriented sweeps
plot_stage4b_results.py                 # Stage 4B plotting utilities
summarize_stage4b_results.py            # Stage 4B result summaries

run_stage5_compare.py                   # Stage 5 final curated comparison
run_stage5_sweep.py                     # Stage 5 sweep experiments
plot_stage5_results.py                  # Stage 5 plotting utilities
summarize_stage5_results.py             # Stage 5 result summaries

run_stage6_compare.py                   # Stage 6 custom CUDA comparison
run_stage6_sweep.py                     # Stage 6 sweep experiments
plot_stage6_results.py                  # Stage 6 plotting utilities
summarize_stage6_results.py             # Stage 6 result summaries

src/common/                             # Device selection and seed utilities
src/benchmark/                          # Timer, CSV utilities, experiment and analysis code
src/attention/sdpa_baseline.py          # Baseline PyTorch SDPA implementation
src/attention/kv_cache.py               # KV-cache utility
src/attention/decode.py                 # Naive and KV-cache decode logic
src/attention/optimized_kv_cache.py     # Stage 4A optimized KV-cache layout
src/attention/optimized_decode.py       # Stage 4A optimized decode path
src/attention/backend_decode.py         # Stage 4B backend decode paths
src/attention/cuda_graph_decode.py      # CUDA Graph replay path
src/attention/stage6_cuda_extension.py  # Custom CUDA extension loading
src/attention/stage6_custom_decode.py   # Stage 6 decode integration
src/attention/cuda/                     # C++/CUDA source files for Stage 6
```

## Setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch matplotlib pyyaml
```

On Euler, load CUDA before running CUDA experiments:

```bash
module purge
module load nvidia/cuda/13.0.0

which nvcc
nvcc --version
nvidia-smi
```

## Example Runs

Run the baseline:

```bash
python run_baseline.py --device cuda --dtype fp16
```

Run the KV-cache comparison:

```bash
python run_kv_cache.py --device cuda --dtype fp16
```

Run the Stage 5 final comparison:

```bash
python run_stage5_compare.py \
  --device cuda \
  --dtype fp16 \
  --prompt-len 512 \
  --gen-steps 32 \
  --heads 8 \
  --head-dim 64
```

Run the Stage 6 custom CUDA comparison:

```bash
export LLM_ATTENTION_STAGE6_TILE_TOKENS=32

python run_stage6_compare.py \
  --device cuda \
  --dtype fp16 \
  --prompt-len 512 \
  --gen-steps 32 \
  --heads 8 \
  --head-dim 64
```

Run the Stage 6 sweep:

```bash
export LLM_ATTENTION_STAGE6_TILE_TOKENS=32

python run_stage6_sweep.py \
  --devices cuda \
  --dtypes fp16 fp32 \
  --prompt-lens 128 256 512 1024 \
  --gen-steps-list 32 \
  --csv-out results/stage6_gpu_sweep_tile32.csv
```

Summarize Stage 6 results:

```bash
python summarize_stage6_results.py \
  --csv results/stage6_gpu_sweep_tile32.csv
```

## Running on Euler with Slurm

Request an interactive GPU node:

```bash
srun --partition=instruction --gres=gpu:1 --time=00:20:00 --pty bash
```

Inside the GPU session:

```bash
module purge
module load nvidia/cuda/13.0.0

source .venv/bin/activate
export LLM_ATTENTION_STAGE6_TILE_TOKENS=32

python run_stage6_compare.py \
  --device cuda \
  --dtype fp16 \
  --prompt-len 512 \
  --gen-steps 32 \
  --heads 8 \
  --head-dim 64
```

## Final Project Report

The final project report and generated plots are included in the `FinalProject/` directory.

The report emphasizes the main project conclusion:

> Algorithmic improvements such as KV caching are essential, but backend/runtime integration can dominate end-to-end GPU decode performance. A minimal custom CUDA kernel is educational and technically meaningful, but it does not automatically outperform mature integrated execution paths such as compiled decode or CUDA Graph replay.

## References

- Vaswani et al., *Attention Is All You Need*, 2017.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
- PyTorch documentation for `scaled_dot_product_attention`
- PyTorch documentation for `torch.compile`
- NVIDIA CUDA Graphs documentation