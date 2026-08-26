#!/usr/bin/env python3
"""Measure one synthetic 257M Tucker microstep without dataset/optimizer noise."""

import argparse
import statistics
import time
from types import SimpleNamespace

import torch

from models.utils import get_model


def make_config(args):
    return SimpleNamespace(
        model="llama",
        vocab_size=50304,
        sequence_length=args.sequence_length,
        n_layer=args.layers,
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
        activation_checkpointing=args.activation_checkpointing,
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
        tucker_forward_mode=args.mode,
        tucker_contract_chunk_size=args.chunk_size,
        tucker_head_contract_chunk_size=args.head_chunk_size,
        tucker_dense_adamw_matrices=True,
        tucker_retract_every_step=False,
        tucker_vector_transport=False,
        tucker_riemannian_muon=False,
        target_parameter_count=257676352 if args.layers == 12 else 0,
        target_parameter_tolerance=12312,
        batch_size=args.batch_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("materialize", "chunked_contract"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--head-chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = get_model(make_config(args)).cuda().train()
    tucker_modules = [
        module for module in model.modules() if hasattr(module, "materialize_weight")
    ]
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError(
            "Invalid benchmark architecture: lm_head must remain dense nn.Linear."
        )
    if len(tucker_modules) != args.layers * 7:
        raise RuntimeError(
            f"Expected {args.layers * 7} internal Tucker modules, got "
            f"{len(tucker_modules)}."
        )
    uncompiled_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if args.layers == 12 and uncompiled_parameter_count != 257676352:
        raise RuntimeError(
            "Invalid 12-layer target parameter count: expected 257676352, got "
            f"{uncompiled_parameter_count}."
        )
    if args.compile:
        model = torch.compile(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (args.batch_size, args.sequence_length),
        device="cuda",
    )
    targets = torch.randint_like(tokens, 0, model.config.vocab_size)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        return float(loss.detach())

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    loss = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated()
    print(
        f"mode={args.mode} layers={args.layers} batch={args.batch_size} "
        f"sequence={args.sequence_length} parameters={parameter_count} "
        f"chunk={args.chunk_size} head_chunk={args.head_chunk_size} "
        f"median_ms={statistics.median(timings):.3f} "
        f"baseline_mib={baseline / 1024**2:.1f} "
        f"peak_mib={peak / 1024**2:.1f} "
        f"step_delta_mib={(peak - baseline) / 1024**2:.1f} loss={loss:.8f}"
    )


if __name__ == "__main__":
    main()
