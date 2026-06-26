#!/usr/bin/env python3
"""Benchmark Muon vs NuMuon optimizer step time on model projection matrices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from third_party.lite.muonlite import MuonLite, NuMuon  # noqa: E402


def mlp_hidden_dim(n_embd: int, multiple_of: int) -> int:
    hidden_dim = int(2 * (4 * n_embd) / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def llama_matrix_specs(n_layer: int, n_embd: int, multiple_of: int):
    hidden = mlp_hidden_dim(n_embd, multiple_of)
    for layer in range(n_layer):
        prefix = f"transformer.h.{layer}"
        yield f"{prefix}.attn.q_proj.weight", (n_embd, n_embd)
        yield f"{prefix}.attn.k_proj.weight", (n_embd, n_embd)
        yield f"{prefix}.attn.v_proj.weight", (n_embd, n_embd)
        yield f"{prefix}.attn.o_proj.weight", (n_embd, n_embd)
        yield f"{prefix}.mlp.gate_proj.weight", (hidden, n_embd)
        yield f"{prefix}.mlp.up_proj.weight", (hidden, n_embd)
        yield f"{prefix}.mlp.down_proj.weight", (n_embd, hidden)


def qwen3_matrix_specs(
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    intermediate_size: int,
):
    q_dim = n_head * head_dim
    kv_dim = n_kv_head * head_dim
    for layer in range(n_layer):
        prefix = f"transformer.h.{layer}"
        yield f"{prefix}.attn.q_proj.weight", (q_dim, n_embd)
        yield f"{prefix}.attn.k_proj.weight", (kv_dim, n_embd)
        yield f"{prefix}.attn.v_proj.weight", (kv_dim, n_embd)
        yield f"{prefix}.attn.o_proj.weight", (n_embd, q_dim)
        yield f"{prefix}.mlp.gate_proj.weight", (intermediate_size, n_embd)
        yield f"{prefix}.mlp.up_proj.weight", (intermediate_size, n_embd)
        yield f"{prefix}.mlp.down_proj.weight", (n_embd, intermediate_size)


def matrix_specs(args):
    if args.architecture == "llama":
        return llama_matrix_specs(args.n_layer, args.n_embd, args.multiple_of)
    if args.architecture == "qwen3":
        return qwen3_matrix_specs(
            args.n_layer,
            args.n_embd,
            args.n_head,
            args.n_kv_head,
            args.head_dim,
            args.intermediate_size,
        )
    raise ValueError(args.architecture)


def make_params(specs, device: str, dtype: torch.dtype):
    params = []
    for name, shape in specs:
        p = torch.empty(shape, device=device, dtype=dtype)
        p.normal_(mean=0.0, std=0.02)
        p.grad = torch.empty_like(p).normal_(mean=0.0, std=0.01)
        params.append((name, p))
    return params


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_optimizer(label: str, opt, params, warmup: int, steps: int, device: str):
    timings = []
    for idx in range(warmup + steps):
        sync(device)
        start = time.perf_counter()
        opt.step()
        sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if idx >= warmup:
            timings.append(elapsed_ms)
    tensor = torch.tensor(timings, dtype=torch.float64)
    return {
        "optimizer": label,
        "steps": steps,
        "warmup": warmup,
        "mean_ms": float(tensor.mean().item()),
        "std_ms": float(tensor.std(unbiased=False).item()),
        "min_ms": float(tensor.min().item()),
        "max_ms": float(tensor.max().item()),
        "num_matrices": len(params),
        "num_parameters": int(sum(p.numel() for _, p in params)),
    }


def build_optimizer(kind: str, params, args):
    common = dict(
        muon_params=params,
        adamw_params=[],
        lr=args.lr,
        weight_decay=args.weight_decay,
        ns_steps=args.ns_steps,
        muon_theta=args.momentum,
        adamw_betas=(0.9, 0.99),
        adamw_eps=1e-8,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
    )
    if kind == "muon":
        return MuonLite(beta1=0.0, beta2=0.0, chi=1.0, chi_adamw=1.0, subspace_ratio=0.0, **common)
    if kind == "numuon":
        return NuMuon(
            numuon_rank_start=args.numuon_rank_start,
            numuon_rank_end=args.numuon_rank_end,
            numuon_rank_scheduler=args.numuon_rank_scheduler,
            numuon_rank_warmup_fraction=args.numuon_rank_warmup_fraction,
            numuon_rank_decay_end_fraction=args.numuon_rank_decay_end_fraction,
            numuon_krylov_iters=args.numuon_krylov_iters,
            numuon_oversample=args.numuon_oversample,
            numuon_warm_start=not args.numuon_no_warm_start,
            **common,
        )
    raise ValueError(kind)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--architecture", default="qwen3", choices=("llama", "qwen3"))
    parser.add_argument("--n-layer", type=int, default=28)
    parser.add_argument("--n-embd", type=int, default=1024)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--n-kv-head", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=6)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--numuon-rank-start", type=float, default=1.0)
    parser.add_argument("--numuon-rank-end", type=float, default=0.25)
    parser.add_argument("--numuon-rank-scheduler", default="cosine", choices=("cosine", "fixed"))
    parser.add_argument("--numuon-rank-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--numuon-rank-decay-end-fraction", type=float, default=0.9)
    parser.add_argument("--numuon-krylov-iters", type=int, default=2)
    parser.add_argument("--numuon-oversample", type=int, default=8)
    parser.add_argument("--numuon-no-warm-start", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    specs = list(matrix_specs(args))

    results = []
    for kind in ("muon", "numuon"):
        torch.manual_seed(0)
        params = make_params(specs, args.device, dtype)
        opt = build_optimizer(kind, params, args)
        result = benchmark_optimizer(kind, opt, params, args.warmup, args.steps, args.device)
        results.append(result)
        print(json.dumps(result, sort_keys=True))
        del opt, params
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"results": results, "args": vars(args)}, indent=2, default=str))


if __name__ == "__main__":
    main()
