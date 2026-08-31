#!/usr/bin/env python3
"""Benchmark one end-to-end Llama training-step configuration on one GPU."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from models.llama import Llama  # noqa: E402
from models.monarch import (  # noqa: E402
    MonarchMuonOptimizer,
    apply_monarch,
    patch_monarch_linear,
)
from optim.sota_opt import dion  # noqa: E402
from scripts.monarch_benchmark.common import (  # noqa: E402
    HARNESS_REVISION,
    atomic_write_json,
    foreign_compute_apps,
    gpu_snapshot,
    model_spec,
    summarize,
    variant_spec,
)


class GPUContaminationError(RuntimeError):
    pass


class ContaminationMonitor:
    """Poll compute-process ownership while warmup and measurement are active."""

    def __init__(self, gpu_uuid: str, allowed_pids: set[int], interval_seconds: float):
        self.gpu_uuid = gpu_uuid
        self.allowed_pids = allowed_pids
        self.interval_seconds = interval_seconds
        self.foreign_processes: list[dict] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-contamination-monitor", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                foreign = foreign_compute_apps(self.gpu_uuid, self.allowed_pids)
            except BaseException as error:
                self.error = f"{type(error).__name__}: {error}"
                self._stop.set()
                return
            if foreign:
                self.foreign_processes = foreign
                self._stop.set()
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        if self._thread.is_alive():
            self.error = self.error or "GPU contamination monitor did not stop"


def make_config(spec: dict, sequence_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        model="llama",
        vocab_size=50304,
        sequence_length=sequence_length,
        dropout=0.0,
        n_layer=spec["n_layer"],
        n_embd=spec["n_embd"],
        n_head=spec["n_head"],
        n_kv_head=0,
        head_dim=0,
        intermediate_size=spec["intermediate_size"],
        multiple_of=256,
        rmsnorm_eps=1e-5,
        rope_theta=10000.0,
        qk_norm=False,
        tie_word_embeddings=False,
        fp8=False,
        qargs=None,
        init_std=0.02,
    )


def instantiate_model(config: SimpleNamespace, device: torch.device) -> Llama:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            return Llama(config)
    finally:
        torch.set_default_dtype(previous_dtype)


def dense_parameter_groups(model: Llama) -> list[dict]:
    by_name = dict(model.named_parameters())
    groups = []
    for specification in model.get_parameter_group_specs():
        group = {key: value for key, value in specification.items() if key != "params"}
        group["params"] = [by_name[name] for name in specification["params"]]
        groups.append(group)
    return groups


def build_model_and_optimizer(args, spec: dict, device: torch.device):
    config = make_config(spec, args.sequence_length)
    model = instantiate_model(config, device)
    model.train()

    if args.variant == "monarch_muon":
        apply_monarch(model, nblocks=args.monarch_blocks, verbose=False)
        patch_monarch_linear(blocked=True, fast_riffle=True)
        muon_params = []
        adamw_params = []
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and not any(
                excluded in name for excluded in ("wte", "lm_head", "embed")
            ):
                muon_params.append(parameter)
            else:
                adamw_params.append(parameter)
        optimizer = MonarchMuonOptimizer(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            momentum=args.momentum,
            adamw_betas=(args.beta1, args.beta2),
            adamw_weight_decay=args.weight_decay,
            adamw_eps=args.eps,
            ns_dtype=torch.bfloat16,
            use_foreach=True,
        )
    elif args.variant == "dense_adamw":
        groups = dense_parameter_groups(model)
        for group in groups:
            group.pop("is_proj_params", None)
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
            fused=True,
        )
    elif args.variant == "dense_muon":
        groups = dense_parameter_groups(model)
        for group in groups:
            if not group.get("is_proj_params", False):
                group["algorithm"] = "adamw"
        optimizer = dion.Muon(
            groups,
            distributed_mesh=None,
            lr=args.lr,
            mu=args.momentum,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=True,
            adjust_lr="rms_norm",
            newton_schulz_func_name="jordan",
            muon_ns_steps=5,
            adamw_lr_scale=1.0,
            dampening=0.0,
        )
    else:
        raise ValueError(args.variant)

    return model, optimizer


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        return local.numel() * local.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def tensor_dtypes(value, *, nonscalar_only: bool = False) -> set[str]:
    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        if nonscalar_only and local.numel() <= 1:
            return set()
        return {str(local.dtype).removeprefix("torch.")}
    if isinstance(value, dict):
        collections = [
            tensor_dtypes(item, nonscalar_only=nonscalar_only)
            for item in value.values()
        ]
        return set().union(*collections) if collections else set()
    if isinstance(value, (tuple, list)):
        collections = [
            tensor_dtypes(item, nonscalar_only=nonscalar_only) for item in value
        ]
        return set().union(*collections) if collections else set()
    return set()


def timed_step(model, optimizer, batches, stream: torch.cuda.Stream) -> dict:
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    forward_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in batches
    ]
    backward_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in batches
    ]
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    host_start = time.perf_counter_ns()
    losses = []
    with torch.cuda.stream(stream):
        total_start.record(stream)
        for index, (inputs, targets) in enumerate(batches):
            forward_pairs[index][0].record(stream)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs, targets=targets)["loss"] / len(batches)
            forward_pairs[index][1].record(stream)
            backward_pairs[index][0].record(stream)
            loss.backward()
            backward_pairs[index][1].record(stream)
            losses.append(loss.detach())

        optimizer_start.record(stream)
        optimizer.step()
        optimizer_end.record(stream)
        optimizer.zero_grad(set_to_none=True)
        total_end.record(stream)

    torch.cuda.synchronize()
    host_total_ms = (time.perf_counter_ns() - host_start) / 1e6
    return {
        "host_total_ms": host_total_ms,
        "gpu_total_ms": total_start.elapsed_time(total_end),
        "forward_ms": sum(start.elapsed_time(end) for start, end in forward_pairs),
        "backward_ms": sum(start.elapsed_time(end) for start, end in backward_pairs),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "loss": sum(float(loss.float().item()) for loss in losses),
    }


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = model_spec(args.model_size)
    variant = variant_spec(args.variant)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    initial_foreign = foreign_compute_apps(args.gpu_uuid, {os.getpid()})
    if initial_foreign:
        raise GPUContaminationError(
            f"GPU became occupied before model construction: {initial_foreign}"
        )

    model, optimizer = build_model_and_optimizer(args, spec, device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = (
        spec["monarch_params_expected"][args.monarch_blocks]
        if args.variant == "monarch_muon"
        else spec["dense_params_expected"]
    )
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"parameter-count mismatch: actual={actual_parameters}, expected={expected_parameters}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    batches = [
        (
            torch.randint(
                0,
                50304,
                (args.microbatch, args.sequence_length),
                generator=generator,
                device=device,
            ),
            torch.randint(
                0,
                50304,
                (args.microbatch, args.sequence_length),
                generator=generator,
                device=device,
            ),
        )
        for _ in range(args.accumulation_steps)
    ]
    tokens_per_step = args.microbatch * args.sequence_length * args.accumulation_steps
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    stream.synchronize()

    allowed_pids = {os.getpid()}
    before_warmup = foreign_compute_apps(args.gpu_uuid, allowed_pids)
    if before_warmup:
        raise GPUContaminationError(f"GPU became occupied before warmup: {before_warmup}")
    monitor = ContaminationMonitor(
        args.gpu_uuid,
        allowed_pids,
        args.contamination_poll_seconds,
    )
    monitor.start()
    try:
        for _ in range(args.warmup_steps):
            sample = timed_step(model, optimizer, batches, stream)
            if not math.isfinite(sample["loss"]):
                raise RuntimeError(f"non-finite warmup loss: {sample['loss']}")

        torch.cuda.reset_peak_memory_stats(device)
        telemetry_before = gpu_snapshot(args.gpu_uuid)
        before_measurement = foreign_compute_apps(args.gpu_uuid, allowed_pids)
        if before_measurement:
            raise GPUContaminationError(
                f"GPU became occupied before measurement: {before_measurement}"
            )

        samples = []
        for iteration in range(args.measured_steps):
            sample = timed_step(model, optimizer, batches, stream)
            if not math.isfinite(sample["loss"]):
                raise RuntimeError(f"non-finite measured loss at {iteration}: {sample['loss']}")
            sample["iteration"] = iteration
            sample["tokens_per_second"] = tokens_per_step / (sample["host_total_ms"] / 1000.0)
            samples.append(sample)
    finally:
        monitor.stop()

    if monitor.error:
        raise GPUContaminationError(f"GPU contamination monitoring failed: {monitor.error}")
    if monitor.foreign_processes:
        raise GPUContaminationError(
            f"GPU was contaminated during warmup or measurement: {monitor.foreign_processes}"
        )

    telemetry_after = gpu_snapshot(args.gpu_uuid)
    after_measurement = foreign_compute_apps(args.gpu_uuid, allowed_pids)
    if after_measurement:
        raise GPUContaminationError(
            f"GPU was contaminated during measurement: {after_measurement}"
        )

    optimizer_state_dtypes = sorted(tensor_dtypes(optimizer.state))
    optimizer_nonscalar_state_dtypes = sorted(
        tensor_dtypes(optimizer.state, nonscalar_only=True)
    )
    if optimizer_nonscalar_state_dtypes != ["bfloat16"]:
        raise RuntimeError(
            "unexpected nonscalar optimizer-state dtypes: "
            f"{optimizer_nonscalar_state_dtypes}"
        )

    metrics = (
        "host_total_ms",
        "gpu_total_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "tokens_per_second",
    )
    summary = {metric: summarize([sample[metric] for sample in samples]) for metric in metrics}
    result = {
        "status": "complete",
        "model": {
            **spec,
            "dense_equivalent_parameters": spec["dense_params_expected"],
            "actual_parameters": actual_parameters,
        },
        "variant": variant,
        "benchmark": {
            "harness_revision": HARNESS_REVISION,
            "storage_dtype": "bfloat16",
            "autocast_dtype": "bfloat16",
            "optimizer_moment_dtype": "bfloat16",
            "sequence_length": args.sequence_length,
            "microbatch": args.microbatch,
            "accumulation_steps": args.accumulation_steps,
            "tokens_per_step": tokens_per_step,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "compile_model": False,
            "monarch_blocks": args.monarch_blocks,
            "lr": args.lr,
            "momentum": args.momentum,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "eps": args.eps,
            "seed": args.seed,
            "contamination_poll_seconds": args.contamination_poll_seconds,
            "adamw_fused": args.variant == "dense_adamw",
            "optimizer_backend": {
                "monarch_muon": "MonarchMuonOptimizer",
                "dense_adamw": "torch.optim.AdamW",
                "dense_muon": "dion.Muon",
            }[args.variant],
            "newton_schulz_steps": None if args.variant == "dense_adamw" else 5,
            "cuda_timing": "events on a dedicated stream; device-wide sync at step end",
        },
        "gpu": {
            "uuid": args.gpu_uuid,
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "foreign_processes_before": before_measurement,
            "foreign_processes_after": after_measurement,
            "contamination_monitor": {
                "poll_seconds": args.contamination_poll_seconds,
                "foreign_processes_seen": monitor.foreign_processes,
                "error": monitor.error,
            },
        },
        "memory": {
            "model_bytes": sum(tensor_bytes(parameter) for parameter in model.parameters()),
            "gradient_bytes_nominal": sum(
                tensor_bytes(parameter) for parameter in model.parameters()
            ),
            "optimizer_state_bytes": tensor_bytes(optimizer.state),
            "optimizer_state_dtypes": optimizer_state_dtypes,
            "optimizer_nonscalar_state_dtypes": optimizer_nonscalar_state_dtypes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "summary": summary,
        "samples": samples,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": os.getpid(),
        },
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("monarch_muon", "dense_adamw", "dense_muon"))
    parser.add_argument("--model-size", required=True, choices=("257m", "834m", "1p4b", "3p5b", "6p9b"))
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--monarch-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contamination-poll-seconds", type=float, default=0.25)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.contamination_poll_seconds <= 0:
        raise ValueError("--contamination-poll-seconds must be positive")
    if min(args.microbatch, args.sequence_length, args.accumulation_steps) <= 0:
        raise ValueError("microbatch, sequence length, and accumulation steps must be positive")
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("warmup and measured steps must both be positive")
    started = time.time()
    try:
        payload = run(args)
        payload["wall_started_unix"] = started
        payload["wall_finished_unix"] = time.time()
        atomic_write_json(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            "model": payload["model"]["name"],
            "variant": payload["variant"]["name"],
            "median_ms": payload["summary"]["host_total_ms"]["median"],
            "tokens_per_second": payload["summary"]["tokens_per_second"]["median"],
            "peak_gb": payload["memory"]["peak_allocated_bytes"] / 1e9,
        }, sort_keys=True))
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            status = "oom"
        elif isinstance(error, GPUContaminationError):
            status = "contaminated"
        else:
            status = "failed"
        payload = {
            "status": status,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "model_size": args.model_size,
            "variant": args.variant,
            "gpu_uuid": args.gpu_uuid,
            "wall_started_unix": started,
            "wall_finished_unix": time.time(),
        }
        atomic_write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
