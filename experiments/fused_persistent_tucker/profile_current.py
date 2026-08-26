#!/usr/bin/env python3
"""Profile the current direct Tucker fwd+bwd path on one CUDA device.

This intentionally does not include an optimizer.  The printed table uses CUDA
device time and is therefore suitable for identifying kernel-level hotspots.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


_HERE = Path(__file__).resolve()
ROOT = next(
    candidate
    for candidate in (_HERE.parents[1], _HERE.parents[2])
    if (candidate / "src" / "models").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(_HERE.parent))

from models.utils import get_model  # noqa: E402


def make_config(args):
    return SimpleNamespace(
        model="llama",
        vocab_size=50304,
        sequence_length=args.sequence_length,
        n_layer=12,
        n_embd=1024,
        n_head=8,
        ffn_hidden_size=2816,
        multiple_of=256,
        dropout=0.0,
        rmsnorm_eps=1e-6,
        init_std=0.02,
        dtype="bfloat16",
        device="cuda",
        fp8=False,
        fp8_optim=False,
        liger_kernels=True,
        activation_checkpointing=False,
        label_smoothing=0.0,
        attention_type="standard",
        qkv_clipping=False,
        linear_parameterization="tucker",
        tucker_rank=259,
        tucker_ranks=None,
        tucker_attention_ranks=None,
        tucker_gate_up_ranks=None,
        tucker_down_ranks=None,
        tucker_rank_plan=None,
        tucker_equal_params=False,
        tucker_forward_mode="chunked_contract",
        tucker_contract_chunk_size=args.chunk_size,
        tucker_head_contract_chunk_size=args.head_chunk_size,
        tucker_dense_adamw_matrices=True,
        tucker_retract_every_step=False,
        tucker_vector_transport=False,
        tucker_riemannian_muon=False,
        target_parameter_count=257676352,
        target_parameter_tolerance=12312,
        batch_size=args.batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--head-chunk-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = get_model(make_config(args)).cuda().train()
    if args.experimental:
        from tucker_fused_ops import install

        install()
    tokens = torch.randint(
        0, 50304, (args.batch_size, args.sequence_length), device="cuda"
    )
    targets = torch.randint_like(tokens, 0, 50304)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        return loss

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

    params = sum(p.numel() for p in model.parameters())
    print(
        f"parameters={params} elapsed_ms={elapsed_ms:.3f} "
        f"loss={float(loss):.8f} gpu={torch.cuda.get_device_name()}"
    )
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.top))
    trace_path = os.environ.get("TUCKER_TRACE_PATH")
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"trace={trace_path}")


if __name__ == "__main__":
    main()
