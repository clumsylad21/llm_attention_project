# src/attention/cuda_graph_decode.py

from typing import Callable, Optional, Tuple

import torch

from src.attention.backend_decode import (
    allocate_backend_decode_buffers,
    run_backend_decode_preallocated,
)
from src.attention.optimized_decode import FusedProjectionWeights


class CUDAGraphBackendDecodeRunner:
    """
    Stage 4B-C: fixed-shape CUDA Graph replay runner.

    Important constraints:
    - GPU only
    - fixed batch / prompt_len / gen_steps / heads / head_dim
    - same input tensor shape every replay
    """

    def __init__(
        self,
        sample_hidden_states: torch.Tensor,
        prompt_len: int,
        gen_steps: int,
        fused_weights: FusedProjectionWeights,
    ) -> None:
        if sample_hidden_states.device.type != "cuda":
            raise ValueError("CUDA Graph runner requires a CUDA tensor.")

        self.prompt_len = prompt_len
        self.gen_steps = gen_steps
        self.fused_weights = fused_weights

        batch = sample_hidden_states.size(0)
        total_seq_len = prompt_len + gen_steps

        self.static_hidden_states = torch.empty_like(sample_hidden_states)

        self.buffers = allocate_backend_decode_buffers(
            batch=batch,
            heads=fused_weights.num_heads,
            head_dim=fused_weights.head_dim,
            total_seq_len=total_seq_len,
            gen_steps=gen_steps,
            device=sample_hidden_states.device,
            dtype=sample_hidden_states.dtype,
        )

        # Warm up on a side stream before capture.
        warmup_stream = torch.cuda.Stream(device=sample_hidden_states.device)
        current_stream = torch.cuda.current_stream(device=sample_hidden_states.device)

        warmup_stream.wait_stream(current_stream)

        with torch.cuda.stream(warmup_stream):
            self.static_hidden_states.copy_(sample_hidden_states)

            for _ in range(3):
                self.static_output = run_backend_decode_preallocated(
                    all_hidden_states=self.static_hidden_states,
                    prompt_len=self.prompt_len,
                    gen_steps=self.gen_steps,
                    fused_weights=self.fused_weights,
                    buffers=self.buffers,
                )

        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(sample_hidden_states.device)

        self.graph = torch.cuda.CUDAGraph()

        # Seed static inputs before capture.
        self.static_hidden_states.copy_(sample_hidden_states)

        with torch.cuda.graph(self.graph):
            self.static_output = run_backend_decode_preallocated(
                all_hidden_states=self.static_hidden_states,
                prompt_len=self.prompt_len,
                gen_steps=self.gen_steps,
                fused_weights=self.fused_weights,
                buffers=self.buffers,
            )

    def __call__(self, all_hidden_states: torch.Tensor) -> torch.Tensor:
        self.static_hidden_states.copy_(all_hidden_states)
        self.graph.replay()
        return self.static_output


def make_cuda_graph_backend_runner(
    sample_hidden_states: torch.Tensor,
    prompt_len: int,
    gen_steps: int,
    fused_weights: FusedProjectionWeights,
    enable_cuda_graphs: bool = True,
) -> Tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], str]:
    """
    Build the Stage 4B-C runner if possible.

    Returns:
        runner_or_none, status_string
    """
    if not enable_cuda_graphs:
        return None, "cuda_graph_disabled"

    if sample_hidden_states.device.type != "cuda":
        return None, "cuda_graph_non_cuda_device"

    if not torch.cuda.is_available():
        return None, "cuda_graph_cuda_unavailable"

    if not hasattr(torch.cuda, "CUDAGraph"):
        return None, "cuda_graph_api_unavailable"

    try:
        runner = CUDAGraphBackendDecodeRunner(
            sample_hidden_states=sample_hidden_states,
            prompt_len=prompt_len,
            gen_steps=gen_steps,
            fused_weights=fused_weights,
        )
        return runner, "cuda_graph_enabled"
    except Exception as exc:
        return None, f"cuda_graph_fallback_{type(exc).__name__}"