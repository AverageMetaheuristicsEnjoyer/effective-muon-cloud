#!/usr/bin/env python3
"""TP=2 systems benchmark: Megatron Dense Muon versus Monarch-Muon.

The dense variants use NVIDIA Megatron Core's TensorParallelMuon. Monarch
keeps features sharded and uses explicit all-to-all calls between factors and
after the second factor, so every residual boundary is in canonical layout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

GEOMETRIES = {
    # Keeps every TP=2 communication and optimizer path while fitting beside
    # active jobs. It is a correctness/instrumentation smoke test, not a
    # representative systems-scaling point.
    "smoke": {"hidden": 256, "layers": 2, "tokens": 16, "heads": 2, "ffn_hidden": 1024},
    "small": {"hidden": 1024, "layers": 4, "tokens": 128, "heads": 8, "ffn_hidden": 2752},
    "medium": {"hidden": 2048, "layers": 4, "tokens": 512, "heads": 16, "ffn_hidden": 5504},
    "large": {"hidden": 4096, "layers": 4, "tokens": 2048, "heads": 32, "ffn_hidden": 11008},
    # Exactly the decoder geometry used by scripts/monarch_benchmark: 6.89B
    # dense parameters, including the vocabulary embedding and output head.
    "llama7b": {"hidden": 4096, "layers": 32, "tokens": 1024, "heads": 32, "ffn_hidden": 11008},
}
DENSE_VARIANTS = ("dense_duplicated", "dense_distributed", "dense_blockwise")
VARIANTS = (*DENSE_VARIANTS, "monarch_muon")
VOCAB_SIZE = 50_304


def world_size() -> int:
    return dist.get_world_size()


def rank() -> int:
    return dist.get_rank()


def tp_group() -> dist.ProcessGroup:
    return dist.group.WORLD


@dataclass
class CommunicationStats:
    """Records CUDA-event time and API-level payloads for actual collectives."""

    enabled: bool = False
    phase: str = "idle"
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    payload_bytes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def reset(self) -> None:
        self.phase = "idle"
        self.events.clear()
        self.payload_bytes.clear()
        self.calls.clear()

    def record(
        self, op: str, tensor: torch.Tensor, fn: Callable[[], Any], *, count_payload: bool = True
    ) -> Any:
        if not self.enabled or self.phase == "idle":
            return fn()
        key = f"{self.phase}:{op}"
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        # Per-rank collective input bytes. This is topology-independent, unlike
        # wire bytes, which vary with NCCL's ring/tree algorithm and NVLink path.
        self.events[key].append((start, end))
        if count_payload:
            self.payload_bytes[key] += tensor.numel() * tensor.element_size()
            self.calls[key] += 1
        return result

    def elapsed_ms(self, prefix: str) -> float:
        return sum(
            start.elapsed_time(end)
            for key, pairs in self.events.items()
            if key.startswith(prefix)
            for start, end in pairs
        )

    def bytes(self, prefix: str) -> int:
        return sum(value for key, value in self.payload_bytes.items() if key.startswith(prefix))

    def breakdown(self) -> dict[str, dict[str, float | int]]:
        """Per-phase collective calls, logical payload, and CUDA-event time."""
        keys = set(self.calls) | set(self.payload_bytes)
        return {
            key: {
                "calls": self.calls[key],
                "bytes": self.payload_bytes[key],
                "ms": sum(start.elapsed_time(end) for start, end in self.events[key]),
            }
            for key in sorted(keys)
        }


STATS = CommunicationStats()
_COLLECTIVES_INSTALLED = False


def _collective_tensor(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    if name in {"all_gather", "all_gather_into_tensor", "_all_gather_base"}:
        return args[1] if len(args) > 1 else kwargs.get("tensor")
    if name in {"all_to_all_single", "reduce_scatter", "reduce_scatter_tensor", "_reduce_scatter_base"}:
        return args[1] if len(args) > 1 else kwargs.get("input")
    return args[0] if args else kwargs.get("tensor")


def install_collective_instrumentation() -> None:
    """Instrument PyTorch collectives without changing Megatron's implementation."""
    global _COLLECTIVES_INSTALLED
    if _COLLECTIVES_INSTALLED:
        return
    for name in (
        "all_reduce", "all_gather", "all_gather_into_tensor", "_all_gather_base",
        "all_to_all_single", "reduce_scatter", "reduce_scatter_tensor", "_reduce_scatter_base",
    ):
        if not hasattr(dist, name):
            continue
        original = getattr(dist, name)

        def wrapped(*args, __name=name, __original=original, **kwargs):
            tensor = _collective_tensor(__name, args, kwargs)
            if not isinstance(tensor, torch.Tensor):
                return __original(*args, **kwargs)
            return STATS.record(__name, tensor, lambda: __original(*args, **kwargs))

        setattr(dist, name, wrapped)
    _COLLECTIVES_INSTALLED = True


def all_to_all(packed: torch.Tensor) -> torch.Tensor:
    """PyTorch's autograd-aware all-to-all used by both Monarch permutations."""
    return dist_nn.all_to_all_single(torch.empty_like(packed), packed, group=tp_group())


def all_gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Gather canonical Monarch feature shards for the dense vocabulary head."""
    return torch.cat(dist_nn.all_gather(tensor, group=tp_group()), dim=-1)


def local_bmm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One local block BMM, matching the two-factor Monarch compute path."""
    return torch.bmm(x.unsqueeze(0), weight.transpose(-1, -2)).squeeze(0)


class TP2MonarchLinear(nn.Module):
    """A rectangular N=2 Monarch linear with canonical input and output shards.

    Rank ``r`` owns factor blocks ``L_r`` and ``R_r``. The first all-to-all
    changes factor ownership; the second converts factor-2 output back to the
    contiguous feature shard required by residuals and the next layer.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features % 4 or out_features % 4:
            raise ValueError("TP=2 Monarch dimensions must be divisible by four")
        self.in_features, self.out_features = in_features, out_features
        self.local_in, self.local_out = in_features // 2, out_features // 2
        middle = min(in_features, out_features) // 2
        # Keep the singleton block dimension: MonarchMuonOptimizer batches it.
        self.left = nn.Parameter(torch.empty(1, middle, self.local_in))
        self.right = nn.Parameter(torch.empty(1, self.local_out, middle))
        bound = 1.0 / math.sqrt(self.local_in)
        nn.init.uniform_(self.left, -bound, bound)
        nn.init.uniform_(self.right, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.local_in)
        first = local_bmm(flat, self.left)
        # [rank-owned L block, coordinate] -> [R-owner, source fragment].
        # The reference butterfly reshapes [coordinate, factor-block], so a
        # rank sends alternating coordinates, rather than a contiguous half.
        packed = first.reshape(flat.shape[0], -1, world_size()).permute(2, 0, 1).contiguous()
        mixed = all_to_all(packed).permute(1, 0, 2).reshape_as(first)
        second = local_bmm(mixed, self.right)
        # Factor-2 blocks are strided in the logical output. Canonicalize them
        # before an RMSNorm, residual add, or the first factor of the next layer.
        packed = second.reshape(flat.shape[0], world_size(), -1).permute(1, 0, 2).contiguous()
        canonical = all_to_all(packed).permute(1, 2, 0).reshape_as(second)
        return canonical.reshape(*shape[:-1], self.local_out)


class ShardedRMSNorm(nn.Module):
    """RMSNorm over a feature vector split across the two TP ranks."""

    def __init__(self, hidden: int, eps: float = 1e-5):
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_sum = x.float().square().sum(dim=-1, keepdim=True)
        global_sum = dist_nn.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=tp_group())
        inv_rms = torch.rsqrt(global_sum / self.hidden + self.eps).to(x.dtype)
        return x * inv_rms * self.weight


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """LLaMA RoPE on locally owned attention heads."""
    _, tokens, _, head_dim = q.shape
    positions = torch.arange(tokens, device=q.device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device) / head_dim))
    angles = torch.outer(positions, frequencies).to(q.dtype)
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]

    def rotate(value: torch.Tensor) -> torch.Tensor:
        even, odd = value[..., ::2], value[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def llama_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Causal Flash/SDPA attention over the TP-local contiguous head range."""
    batch, tokens, local_hidden = q.shape
    head_dim = local_hidden // heads
    q = q.view(batch, tokens, heads, head_dim)
    k = k.view(batch, tokens, heads, head_dim)
    v = v.view(batch, tokens, heads, head_dim)
    q, k = apply_rope(q, k)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
    )
    return output.transpose(1, 2).contiguous().view(batch, tokens, local_hidden)


def attention_surrogate(qkv: torch.Tensor) -> torch.Tensor:
    """Legacy helper retained for the CPU TP/Muon regression test."""
    q, k, v = qkv.chunk(3, dim=-1)
    return F.silu(q) * torch.sigmoid(k) + v


class MonarchBlock(nn.Module):
    def __init__(self, hidden: int, ffn_hidden: int, heads: int):
        super().__init__()
        self.norm1 = ShardedRMSNorm(hidden)
        self.local_heads = heads // world_size()
        self.q_proj = TP2MonarchLinear(hidden, hidden)
        self.k_proj = TP2MonarchLinear(hidden, hidden)
        self.v_proj = TP2MonarchLinear(hidden, hidden)
        self.out = TP2MonarchLinear(hidden, hidden)
        self.norm2 = ShardedRMSNorm(hidden)
        self.gate = TP2MonarchLinear(hidden, ffn_hidden)
        self.up = TP2MonarchLinear(hidden, ffn_hidden)
        self.down = TP2MonarchLinear(ffn_hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attention = llama_attention(
            self.q_proj(normed), self.k_proj(normed), self.v_proj(normed), self.local_heads,
        )
        x = x + self.out(attention)
        normed = self.norm2(x)
        return x + self.down(F.silu(self.gate(normed)) * self.up(normed))


class MonarchModel(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int, ffn_hidden: int, config: Any):
        super().__init__()
        self.hidden = hidden
        self.local_hidden = hidden // 2
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding

        common = {"config": config, "init_method": init_method, "bias": False, "tp_group": tp_group()}
        self.embedding = VocabParallelEmbedding(
            VOCAB_SIZE, hidden, init_method=init_method, config=config, tp_group=tp_group(),
        )
        self.blocks = nn.ModuleList(MonarchBlock(hidden, ffn_hidden, heads) for _ in range(layers))
        self.final_norm = ShardedRMSNorm(hidden)
        self.lm_head = ColumnParallelLinear(hidden, VOCAB_SIZE, gather_output=False, **common)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).chunk(world_size(), dim=-1)[rank()].contiguous()
        for block in self.blocks:
            x = block(x)
        logits, _ = self.lm_head(all_gather_last_dim(self.final_norm(x)))
        from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

        return vocab_parallel_cross_entropy(logits, targets).mean()


def megatron_config() -> Any:
    from megatron.core.model_parallel_config import ModelParallelConfig

    return ModelParallelConfig(
        tensor_model_parallel_size=world_size(), params_dtype=torch.bfloat16,
        perform_initialization=True, sequence_parallel=False, gradient_accumulation_fusion=False,
    )


def init_method(weight: torch.Tensor) -> None:
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class DenseBlock(nn.Module):
    """Megatron Core TP projection layout for a Llama-like transformer block."""

    def __init__(self, hidden: int, ffn_hidden: int, heads: int, config: Any):
        super().__init__()
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

        common = {"config": config, "init_method": init_method, "bias": False, "tp_group": tp_group()}
        self.norm1 = nn.RMSNorm(hidden, eps=1e-5)
        self.local_heads = heads // world_size()
        self.q_proj = ColumnParallelLinear(hidden, hidden, gather_output=False, **common)
        self.k_proj = ColumnParallelLinear(hidden, hidden, gather_output=False, **common)
        self.v_proj = ColumnParallelLinear(hidden, hidden, gather_output=False, **common)
        self.out = RowParallelLinear(hidden, hidden, input_is_parallel=True, skip_bias_add=False, **common)
        self.norm2 = nn.RMSNorm(hidden, eps=1e-5)
        self.gate = ColumnParallelLinear(hidden, ffn_hidden, gather_output=False, **common)
        self.up = ColumnParallelLinear(hidden, ffn_hidden, gather_output=False, **common)
        self.down = RowParallelLinear(ffn_hidden, hidden, input_is_parallel=True, skip_bias_add=False, **common)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        q, _ = self.q_proj(normed)
        k, _ = self.k_proj(normed)
        v, _ = self.v_proj(normed)
        attn, _ = self.out(llama_attention(q, k, v, self.local_heads))
        x = x + attn
        normed = self.norm2(x)
        gate, _ = self.gate(normed)
        up, _ = self.up(normed)
        mlp, _ = self.down(F.silu(gate) * up)
        return x + mlp


class DenseModel(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int, ffn_hidden: int):
        super().__init__()
        config = megatron_config()
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding

        self.hidden = hidden
        common = {"config": config, "init_method": init_method, "bias": False, "tp_group": tp_group()}
        self.embedding = VocabParallelEmbedding(
            VOCAB_SIZE, hidden, init_method=init_method, config=config, tp_group=tp_group(),
        )
        self.blocks = nn.ModuleList(DenseBlock(hidden, ffn_hidden, heads, config) for _ in range(layers))
        self.final_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.lm_head = ColumnParallelLinear(hidden, VOCAB_SIZE, gather_output=False, **common)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        logits, _ = self.lm_head(self.final_norm(x))
        from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

        return vocab_parallel_cross_entropy(logits, targets).mean()


class CompositeOptimizer:
    """One optimizer interface for Muon projection weights and AdamW norm weights."""

    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer):
        self.muon, self.adamw = muon, adamw

    @torch.no_grad()
    def step(self) -> None:
        self.muon.step()
        self.adamw.step()


class MasterTensorParallelMuon:
    """Run MCore Muon on FP32 master weights for a BF16 TP model."""

    def __init__(self, parameters: list[nn.Parameter], **kwargs: Any):
        from megatron.core.optimizer.muon import TensorParallelMuon

        self.model_parameters = parameters
        self.master_parameters: list[nn.Parameter] = []
        for parameter in parameters:
            master = nn.Parameter(parameter.detach().float().clone())
            # TensorParallelMuon reads TP metadata from the parameter itself.
            for attribute in ("tensor_model_parallel", "partition_dim", "partition_stride"):
                if hasattr(parameter, attribute):
                    setattr(master, attribute, getattr(parameter, attribute))
            self.master_parameters.append(master)
        self.optimizer = TensorParallelMuon(self.master_parameters, **kwargs)

    @torch.no_grad()
    def step(self) -> None:
        for parameter, master in zip(self.model_parameters, self.master_parameters):
            master.grad = parameter.grad.detach().float()
        self.optimizer.step()
        for parameter, master in zip(self.model_parameters, self.master_parameters):
            parameter.copy_(master.to(dtype=parameter.dtype))


def matrix_and_scalar_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    matrices, scalars = [], []
    for name, parameter in model.named_parameters():
        # Match the single-GPU benchmark: embeddings and lm_head use AdamW.
        if parameter.ndim >= 2 and "embedding" not in name and "lm_head" not in name:
            matrices.append(parameter)
        else:
            scalars.append(parameter)
    return matrices, scalars


def build_optimizer(model: nn.Module, variant: str, args: argparse.Namespace) -> CompositeOptimizer:
    matrices, scalars = matrix_and_scalar_params(model)
    adamw = torch.optim.AdamW(
        scalars, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
        fused=True,
    )
    if variant == "monarch_muon":
        from models.monarch.monarch_muon import MonarchMuonOptimizer

        muon = MonarchMuonOptimizer(
            matrices, [], lr=args.lr, momentum=args.momentum, ns_dtype=torch.bfloat16,
            use_foreach=True,
        )
        return CompositeOptimizer(muon, adamw)

    from megatron.core.process_groups_config import ProcessGroupCollection

    muon = MasterTensorParallelMuon(
        matrices, lr=args.lr, momentum_beta=args.momentum, use_nesterov=True,
        weight_decay=args.weight_decay, num_ns_steps=5, scale_mode="spectral",
        pg_collection=ProcessGroupCollection(tp=tp_group()), mode=variant.removeprefix("dense_"),
    )
    return CompositeOptimizer(muon, adamw)


def install_ns_instrumentation() -> None:
    """Time the real NVIDIA NS implementation and the project Monarch NS kernels."""
    import megatron.core.optimizer.muon as megatron_muon
    import models.monarch.monarch_muon as monarch_muon

    def timed(module: Any, name: str) -> None:
        original = getattr(module, name)
        if getattr(original, "_tp2_timed", False):
            return

        def wrapped(*args, **kwargs):
            return STATS.record(
                "newton_schulz", args[0], lambda: original(*args, **kwargs), count_payload=False,
            )

        wrapped._tp2_timed = True
        setattr(module, name, wrapped)

    timed(megatron_muon, "newton_schulz_tp")
    timed(monarch_muon, "_newton_schulz")
    timed(monarch_muon, "_newton_schulz_batched")


def reset_stats() -> None:
    STATS.reset()
    STATS.enabled = True


def max_rank_value(value: float | int, device: torch.device) -> float:
    STATS.enabled = False
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=tp_group())
    STATS.enabled = True
    return float(tensor.item())


def max_rank_collective_breakdown() -> dict[str, dict[str, float | int]]:
    """Report the slowest rank's payload/time for every measured collective."""
    local = STATS.breakdown()
    gathered: list[dict[str, dict[str, float | int]] | None] = [None] * world_size()
    STATS.enabled = False
    dist.all_gather_object(gathered, local, group=tp_group())
    STATS.enabled = True
    keys = set().union(*(item.keys() for item in gathered if item is not None))
    return {
        key: {
            "calls": max(int(item.get(key, {}).get("calls", 0)) for item in gathered if item is not None),
            "bytes": max(int(item.get(key, {}).get("bytes", 0)) for item in gathered if item is not None),
            "ms": max(float(item.get(key, {}).get("ms", 0.0)) for item in gathered if item is not None),
        }
        for key in sorted(keys)
    }


def run_step(
    model: nn.Module,
    optimizer: CompositeOptimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict:
    reset_stats()
    torch.cuda.reset_peak_memory_stats(device)
    start, forward_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    backward_end, optimizer_start = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    optimizer_end, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    STATS.phase = "forward"
    loss = model(tokens, targets)
    forward_end.record()
    STATS.phase = "backward"
    loss.backward()
    backward_end.record()
    STATS.phase = "optimizer"
    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    model.zero_grad(set_to_none=True)
    STATS.phase = "idle"
    end.record()
    torch.cuda.synchronize(device)
    local = {
        "step_ms": start.elapsed_time(end),
        "forward_ms": start.elapsed_time(forward_end),
        "backward_ms": forward_end.elapsed_time(backward_end),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "newton_schulz_ms": STATS.elapsed_ms("optimizer:newton_schulz"),
        "activation_nccl_ms": STATS.elapsed_ms("forward:") + STATS.elapsed_ms("backward:"),
        "activation_nccl_bytes": STATS.bytes("forward:") + STATS.bytes("backward:"),
        "optimizer_nccl_ms": STATS.elapsed_ms("optimizer:") - STATS.elapsed_ms("optimizer:newton_schulz"),
        "optimizer_nccl_bytes": STATS.bytes("optimizer:"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    result = {key: max_rank_value(value, device) for key, value in local.items()}
    result["collective_breakdown"] = max_rank_collective_breakdown()
    return result


def gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(world_size())]
    dist.all_gather(gathered, tensor, group=tp_group())
    return torch.cat(gathered, dim=-1)


@torch.no_grad()
def validate_monarch(model: MonarchModel, device: torch.device) -> float:
    """Compare a distributed rectangular N=2 projection to the repository kernel."""
    from models.monarch.monarch_linear import blockdiag_butterfly_multiply

    linear = model.blocks[0].q_proj
    tokens = 3
    # Every TP rank needs the same logical input for the reference comparison.
    torch.manual_seed(991)
    full = torch.randn(tokens, linear.in_features, device=device, dtype=torch.bfloat16)
    local = full.chunk(world_size(), dim=-1)[rank()].contiguous()
    actual = gather_last_dim(linear(local))
    left = [torch.empty_like(linear.left) for _ in range(world_size())]
    right = [torch.empty_like(linear.right) for _ in range(world_size())]
    dist.all_gather(left, linear.left, group=tp_group())
    dist.all_gather(right, linear.right, group=tp_group())
    expected = blockdiag_butterfly_multiply(full, torch.cat(left), torch.cat(right))
    return max_rank_value(float((actual.float() - expected.float()).abs().max()), device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=GEOMETRIES, default="small")
    parser.add_argument("--variant", choices=VARIANTS, default="monarch_muon")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--measured-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ffn-hidden", type=int, default=None)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--validate-monarch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def setup_model_parallel() -> None:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size())
    # GPU-initialized MCore TP layers draw weights through this tracker.  It is
    # distinct from torch.manual_seed and must exist before their construction.
    model_parallel_cuda_manual_seed(17)


def summarize(samples: list[dict]) -> dict:
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in samples[0]
        if isinstance(samples[0][key], (float, int))
    }


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    if world_size() != 2:
        raise RuntimeError(f"TP=2 benchmark requires exactly two ranks, got {world_size()}")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    setup_model_parallel()
    install_collective_instrumentation()
    install_ns_instrumentation()
    spec = GEOMETRIES[args.geometry]
    tokens = args.tokens or spec["tokens"]
    layers = args.layers or spec["layers"]
    ffn_hidden = args.ffn_hidden or spec["ffn_hidden"]
    torch.manual_seed(17 + rank())
    if args.variant == "monarch_muon":
        model: nn.Module = MonarchModel(spec["hidden"], layers, spec["heads"], ffn_hidden, megatron_config())
    else:
        model = DenseModel(spec["hidden"], layers, spec["heads"], ffn_hidden)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, args.variant, args)
    input_tokens = torch.randint(VOCAB_SIZE, (args.batch_size, tokens), device=device)
    target_tokens = torch.randint(VOCAB_SIZE, (args.batch_size, tokens), device=device)
    validation_error = None
    if args.validate_monarch and args.variant == "monarch_muon":
        validation_error = validate_monarch(model, device)
        if validation_error > 2e-2:
            raise RuntimeError(f"distributed Monarch validation failed: max error {validation_error}")
    for _ in range(args.warmup_steps):
        run_step(model, optimizer, input_tokens, target_tokens, device)
    samples = [run_step(model, optimizer, input_tokens, target_tokens, device) for _ in range(args.measured_steps)]
    if rank() == 0:
        result = {
            "variant": args.variant,
            "geometry": {
                "hidden": spec["hidden"],
                "layers": layers,
                "heads": spec["heads"],
                "ffn_hidden": ffn_hidden,
                "tokens": tokens,
                "batch_size": args.batch_size,
            },
            "world_size": world_size(),
            "median": summarize(samples),
            "collective_breakdown": samples[0]["collective_breakdown"],
            "samples": samples,
            "validation_max_abs_error": validation_error,
            "implementation": {
                "model": "LLaMA decoder: RoPE, causal SDPA, SwiGLU, vocab-parallel embedding/head",
                "dense": "Megatron Core TP layers + TensorParallelMuon",
                "dense_muon_mode": None if args.variant == "monarch_muon" else args.variant.removeprefix("dense_"),
                "monarch": "local BMM -> all_to_all -> local BMM -> all_to_all canonical layout",
                "payload_bytes": "per-rank collective input bytes; not topology-specific wire bytes",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "median": result["median"]}, indent=2), flush=True)
    dist.barrier(group=tp_group())
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
