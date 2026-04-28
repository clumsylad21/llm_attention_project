import statistics
import time
from dataclasses import dataclass

import torch

from src.common.device import sync_if_needed


@dataclass
class BenchResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    checksum: float


def benchmark_function(fn, warmup: int, iters: int, device: str) -> BenchResult:
    # Warmup runs
    for _ in range(warmup):
        out = fn()
    sync_if_needed(device)

    times_ms = []

    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            start_event.record()
            out = fn()
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    checksum = float(out.float().mean().item())

    return BenchResult(
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        checksum=checksum,
    )