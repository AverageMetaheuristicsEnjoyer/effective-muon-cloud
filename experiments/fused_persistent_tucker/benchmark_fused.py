#!/usr/bin/env python3
"""Benchmark experimental Tucker operators with parity and memory checks."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
ROOT = next(
    candidate
    for candidate in (HERE.parents[1], HERE.parents[2])
    if (candidate / "src" / "models").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))

from scripts.benchmark_full_tucker_step import make_config  # noqa: E402
from models.utils import get_model  # noqa: E402
from tucker_fused_ops import clear_work_caches, install  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--head-chunk-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--cold-cache", action="store_true",
        help="invalidate BF16 parameter cache before every measured step",
    )
    parser.add_argument(
        "--cuda-graph", action="store_true",
        help="capture the fixed-shape forward+backward and benchmark replay",
    )
    parser.add_argument(
        "--fused-backward", action="store_true",
        help="enable the experimental paired-mode analytical backward",
    )
    parser.add_argument(
        "--online-ce", action="store_true",
        help="use vocabulary-tiled Tucker CE without full chunk logits",
    )
    parser.add_argument("--output-mode-tile", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    config_args = argparse.Namespace(
        mode="chunked_contract",
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        layers=12,
        chunk_size=args.chunk_size,
        head_chunk_size=args.head_chunk_size,
        activation_checkpointing=False,
    )
    model = get_model(make_config(config_args)).cuda().train()
    install(
        fused_backward=args.fused_backward,
        online_ce=args.online_ce,
        output_mode_tile=args.output_mode_tile,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokens = torch.randint(
        0, 50304, (args.batch_size, args.sequence_length), device="cuda"
    )
    targets = torch.randint_like(tokens, 0, 50304)

    def step(clear=False):
        model.zero_grad(set_to_none=True)
        if clear:
            clear_work_caches(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        return loss

    graph = None
    static_loss = None
    if args.cuda_graph:
        if args.cold_cache:
            raise ValueError("--cuda-graph and --cold-cache are mutually exclusive")
        # CUDA graph capture requires all lazy library work and gradient
        # allocations to have happened on a side stream first.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(2, args.warmup)):
                step()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.zero_grad(set_to_none=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                static_loss = model(tokens, targets)["loss"]
            static_loss.backward()
        for _ in range(args.warmup):
            graph.replay()
    else:
        for _ in range(args.warmup):
            step(clear=args.cold_cache)
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    loss = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        if graph is not None:
            graph.replay()
            loss = static_loss
        else:
            loss = step(clear=args.cold_cache)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated()
    print(
        f"implementation=fused_mode_saved_work cold_cache={args.cold_cache} "
        f"cuda_graph={args.cuda_graph} "
        f"fused_backward={args.fused_backward} online_ce={args.online_ce} "
        f"output_mode_tile={args.output_mode_tile} "
        f"batch={args.batch_size} sequence={args.sequence_length} "
        f"parameters={parameter_count} median_ms={statistics.median(timings):.3f} "
        f"all_ms={','.join(f'{value:.3f}' for value in timings)} "
        f"baseline_mib={baseline / 1024**2:.1f} peak_mib={peak / 1024**2:.1f} "
        f"step_delta_mib={(peak-baseline) / 1024**2:.1f} loss={float(loss):.8f}"
    )


if __name__ == "__main__":
    main()
