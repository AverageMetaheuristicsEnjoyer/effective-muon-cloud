import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.llama import Llama


def make_config(args):
    d, ff, heads, layers, vocab = (32, 64, 4, 2, 128) if args.tiny else (1024, 2816, 8, 12, 50304)
    rd, rh, rff = [max(1, int(dim * args.rank_fraction)) for dim in (d, d // heads, ff)]
    return SimpleNamespace(
        vocab_size=vocab, sequence_length=args.sequence_length,
        n_embd=d, n_head=heads, n_layer=layers, ffn_hidden_size=ff,
        dropout=0.0, init_std=0.02, rmsnorm_eps=1e-5, multiple_of=256,
        dtype="bfloat16" if args.device == "cuda" else "float32",
        device=args.device, liger_kernels=False, fp8=False,
        activation_checkpointing=False,
        layerwise_tucker_variant=None if args.variant == "dense" else args.variant,
        layerwise_attention_ranks=(rd, rh, 4),
        layerwise_mlp_ranks=(rff, rd, 2 if args.variant == "A" else 3),
    )


def measure(args):
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.init()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = make_config(args)
    if args.tokens_per_step % (args.microbatch * args.sequence_length):
        raise ValueError("tokens-per-step must be divisible by microbatch * sequence-length")
    accumulation = args.tokens_per_step // (args.microbatch * args.sequence_length)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.device(args.device):
        model = Llama(config).train()
    if cuda:
        torch.cuda.synchronize()
    initialization_ms = 1000 * (time.perf_counter() - start)
    initialization_peak = torch.cuda.max_memory_allocated() if cuda else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, fused=cuda)
    generator = torch.Generator(device=args.device).manual_seed(20260917)
    x = torch.randint(config.vocab_size, (accumulation, args.microbatch, args.sequence_length),
                      generator=generator, device=args.device)
    targets = torch.randint(config.vocab_size, x.shape, generator=generator, device=args.device)

    def step(record):
        events = []
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        losses = []
        for i in range(accumulation):
            if record and cuda:
                f, b, e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                f.record()
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=cuda):
                loss = model(x[i], targets[i])["loss"] / accumulation
            if record and cuda:
                b.record()
            loss.backward()
            if record and cuda:
                e.record()
                events.append((f, b, e))
            losses.append(loss.detach())
        if record and cuda:
            c, o, z, end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            c.record()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if record and cuda:
            o.record()
        optimizer.step()
        if record and cuda:
            z.record()
        optimizer.zero_grad(set_to_none=True)
        if record and cuda:
            end.record()
        if cuda:
            torch.cuda.synchronize()
        host_ms = 1000 * (time.perf_counter() - start)
        value = sum(losses).item()
        if not math.isfinite(value) or not torch.isfinite(norm).item():
            raise RuntimeError("non-finite loss or gradients")
        result = {"host_step_ms": host_ms, "loss": value}
        if record and cuda:
            result.update(
                forward_ms=sum(f.elapsed_time(b) for f, b, _ in events),
                backward_ms=sum(b.elapsed_time(e) for _, b, e in events),
                clip_ms=c.elapsed_time(o), optimizer_ms=o.elapsed_time(z),
                zero_grad_ms=z.elapsed_time(end),
                tokens_per_second=args.tokens_per_step / (host_ms / 1000),
            )
        return result

    warmup = [step(False) for _ in range(args.warmup)]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rows = [step(True) for _ in range(args.steps)]
    summary = {}
    for key in rows[0]:
        values = sorted(row[key] for row in rows)
        summary[key] = {"median": statistics.median(values), "mean": statistics.mean(values),
                        "min": values[0], "max": values[-1]}
    return dict(
        status="complete", args=vars(args), config=vars(config),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
        gpu=torch.cuda.get_device_name() if cuda else None,
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_shapes={name: list(p.shape) for name, p in model.named_parameters()},
        initialization_ms=initialization_ms, initialization_peak_allocated_bytes=initialization_peak,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
        accumulation=accumulation, warmup=warmup, samples=rows, summary=summary,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dense", "A", "B"), default="dense")
    parser.add_argument("--rank-fraction", type=float, choices=(0.25, 0.5), default=0.5)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--tokens-per-step", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.microbatch, args.sequence_length, args.tokens_per_step, args.warmup, args.steps) <= 0:
        parser.error("batch, sequence, tokens, warmup and steps must be positive")
    try:
        result = measure(args)
    except Exception:
        result = {"status": "failed", "args": vars(args), "error": traceback.format_exc()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "args", "parameters", "initialization_ms", "summary", "error") if key in result}), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
