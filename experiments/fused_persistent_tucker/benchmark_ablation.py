#!/usr/bin/env python3
"""Interleaved full-model benchmark for Tucker kernel ablations.

All variants share one initialized model and are alternated by round, reducing
bias from slow model initialization and changing load on a shared A100 host.
"""

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


# The production architecture keeps lm_head dense.  Online Tucker CE is not a
# valid ablation for it, so all published comparisons vary only the internal
# Tucker backward.
VARIANTS = (
    ("baseline_dense_head", False, False),
    ("fused_backward_dense_head", True, False),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--head-chunk-size", type=int, default=2048)
    parser.add_argument("--output-mode-tile", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tucker_module_count = sum(
        hasattr(module, "materialize_weight") for module in model.modules()
    )
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError("lm_head must remain dense nn.Linear")
    if tucker_module_count != 84 or parameter_count != 257676352:
        raise RuntimeError(
            "Invalid target architecture: expected 84 Tucker modules and "
            f"257676352 parameters, got {tucker_module_count} and {parameter_count}."
        )
    tokens = torch.randint(
        0, 50304, (args.batch_size, args.sequence_length), device="cuda"
    )
    targets = torch.randint_like(tokens, 0, 50304)

    def configure(fused_backward, online_ce):
        install(
            fused_backward=fused_backward,
            online_ce=online_ce,
            output_mode_tile=args.output_mode_tile,
        )

    def step(fused_backward, online_ce):
        configure(fused_backward, online_ce)
        model.zero_grad(set_to_none=True)
        clear_work_caches(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        return loss

    for _, fused_backward, online_ce in VARIANTS:
        for _ in range(args.warmup):
            step(fused_backward, online_ce)
    torch.cuda.synchronize()

    timings = {name: [] for name, _, _ in VARIANTS}
    peaks = {name: [] for name, _, _ in VARIANTS}
    losses = {}
    for round_index in range(args.rounds):
        order = VARIANTS if round_index % 2 == 0 else tuple(reversed(VARIANTS))
        for name, fused_backward, online_ce in order:
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            loss = step(fused_backward, online_ce)
            torch.cuda.synchronize()
            timings[name].append((time.perf_counter() - begin) * 1000)
            peaks[name].append(torch.cuda.max_memory_allocated() / 1024**2)
            losses[name] = float(loss)

    print(
        f"parameters={parameter_count} batch={args.batch_size} "
        f"sequence={args.sequence_length} output_mode_tile={args.output_mode_tile}"
    )
    for name, _, _ in VARIANTS:
        print(
            f"variant={name} median_ms={statistics.median(timings[name]):.3f} "
            f"all_ms={','.join(f'{value:.3f}' for value in timings[name])} "
            f"peak_mib={max(peaks[name]):.1f} loss={losses[name]:.8f}"
        )


if __name__ == "__main__":
    main()
