#!/usr/bin/env python3
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from models.tucker_chunked import (
    chunked_tucker_cross_entropy,
    chunked_tucker_linear,
)
from models.tucker_linear import TuckerLinear


SHAPES = {
    "attn": (1024, 1024),
    "up": (1024, 2816),
    "down": (2816, 1024),
    "head": (1024, 50304),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=SHAPES, default="head")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--ce", action="store_true")
    parser.add_argument(
        "--mode", choices=("both", "materialize", "chunked"), default="both"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    in_features, out_features = SHAPES[args.layer]
    module = TuckerLinear(
        in_features,
        out_features,
        rank=259,
        bias=False,
        equal_params=False,
        forward_mode="materialize",
        expected_tokens_per_forward=args.tokens,
        contract_chunk_size=args.chunk_size,
    ).cuda()
    targets = torch.randint(0, out_features, (args.tokens,), device="cuda")

    def run(mode):
        module.zero_grad(set_to_none=True)
        x = torch.randn(
            args.tokens,
            in_features,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.ce:
                if mode == "materialize":
                    weight = module.materialize_weight(dtype=x.dtype)
                    loss = F.cross_entropy(F.linear(x, weight), targets)
                else:
                    loss = chunked_tucker_cross_entropy(
                        x, targets, module, args.chunk_size
                    )
            else:
                if mode == "materialize":
                    weight = module.materialize_weight(dtype=x.dtype)
                    output = F.linear(x, weight)
                else:
                    output = chunked_tucker_linear(x, module, args.chunk_size)
                loss = output.float().square().mean()
        loss.backward()
        return float(loss.detach())

    modes = ("materialize", "chunked") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for _ in range(args.warmup):
            run(mode)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        loss = None
        for _ in range(args.iterations):
            start = time.perf_counter()
            loss = run(mode)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"mode={mode} layer={args.layer} tokens={args.tokens} "
            f"chunk={args.chunk_size} median_ms={statistics.median(timings):.3f} "
            f"peak_mib={peak:.1f} loss={loss:.8f}"
        )


if __name__ == "__main__":
    main()
