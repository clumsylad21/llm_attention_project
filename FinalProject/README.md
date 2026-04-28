# LLM Inference Optimization: Causal Attention and KV Caching

> Performance optimization for decoder-style LLM attention inference using KV caching, PyTorch execution-path improvements, and GPU backend acceleration.

![Python](https://img.shields.io/badge/language-Python-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![CUDA](https://img.shields.io/badge/accelerated_with-CUDA-green)
![Status](https://img.shields.io/badge/status-implemented-success)

## Overview

This repository explores practical performance optimization techniques for decoder-style LLM attention during autoregressive generation.

The project focuses on improving decode-time efficiency by comparing naive attention recomputation, KV-cache-based decoding, and optimized GPU execution paths built with fused projections, cache-aware layouts, `torch.compile`, and CUDA Graph replay.

Rather than treating inference as a single black box, the repository isolates and evaluates the major sources of latency in decoder-style attention, with an emphasis on both implementation clarity and measurable performance improvement.

## Motivation

In decoder-only large language models, token generation happens one step at a time.  
Even when KV caching removes redundant attention recomputation, GPU decode latency can still remain high due to runtime and launch overhead.

This project studies that gap directly by answering questions such as:

- How much does KV caching help compared to naive recomputation?
- What parts of decode latency remain after algorithmic optimization?
- How much additional speedup can be obtained from compiler and runtime techniques?

## Implemented Work

This repository includes:

- a modular PyTorch prototype for decoder-style causal attention
- prefill vs. decode benchmarking
- naive recomputation vs. KV-cache comparison
- reusable CPU/GPU benchmarking pipelines
- correctness checks across execution paths
- automated sweeps with CSV logging and plotting
- optimized decode paths using fused QKV projection and decode-friendly KV-cache layouts
- backend comparisons across eager execution, `torch.compile`, and CUDA Graph replay

## Results Snapshot

Current experiments show that backend/runtime overhead remains a major contributor to GPU decode latency even after introducing KV caching.

Key observations from the implemented benchmark pipeline:

- `torch.compile` provides approximately **1.8–2.0x mean GPU speedup** over baseline KV-cache decode in current sweeps
- explicit **CUDA Graph replay provides ~5x+ mean GPU speedup**, with runs showing **up to ~5.5x speedup** over baseline KV-cache decode

These results highlight an important systems insight:  
**algorithmic improvements such as KV caching help, but backend execution strategy can still dominate end-to-end decode performance.**

## Repository Layout

```text
run_kv_cache.py               # Baseline KV-cache decode execution
run_kv_sweep.py               # Benchmark sweeps, CSV generation, and plotting
run_stage4_compare.py         # Optimized decode path comparison
run_stage4_sweep.py           # Optimized decode sweeps
run_stage4b_compare.py        # Eager vs compile vs CUDA Graph comparison
run_stage4b_sweep.py          # Backend-oriented benchmark sweeps
summarize_stage4b_results.py  # Result aggregation and summary utilities