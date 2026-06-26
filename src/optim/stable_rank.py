"""Stable-rank monitoring utilities for matrix-valued model weights."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import torch


_PROJECTION_GROUPS = (
    ("attn_q", "q_proj.weight"),
    ("attn_k", "k_proj.weight"),
    ("attn_v", "v_proj.weight"),
    ("attn_o", "o_proj.weight"),
    ("ffn_gate", "gate_proj.weight"),
    ("ffn_up", "up_proj.weight"),
    ("ffn_down", "down_proj.weight"),
    ("attn_qkv", "c_attn.weight"),
    ("attn_o", "c_proj.weight"),
    ("ffn_up", "c_fc.weight"),
)


def _projection_group(name: str) -> str | None:
    for group, suffix in _PROJECTION_GROUPS:
        if name.endswith(suffix):
            if suffix == "c_proj.weight" and ".attn." not in name:
                return "ffn_down"
            return group
    return None


def _stable_rank(matrix: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    matrix = matrix.detach().float()
    fro_sq = torch.sum(matrix * matrix)
    if fro_sq <= 0:
        return 0.0, 0.0

    spectral = torch.linalg.matrix_norm(matrix, ord=2)
    stable_rank = fro_sq / (spectral * spectral + eps)
    normalized = stable_rank / min(matrix.shape[-2:])
    return stable_rank.item(), normalized.item()


@torch.no_grad()
def collect_stable_rank(model) -> dict[str, dict[str, float]]:
    """Collect per-projection stable-rank summaries from a non-DDP model."""
    raw_values = defaultdict(list)
    norm_values = defaultdict(list)

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        group = _projection_group(name)
        if group is None:
            continue
        raw, norm = _stable_rank(param)
        raw_values[group].append(raw)
        norm_values[group].append(norm)
        raw_values["all_projections"].append(raw)
        norm_values["all_projections"].append(norm)

    summaries = {}
    for group in sorted(norm_values):
        norm_tensor = torch.tensor(norm_values[group], dtype=torch.float64)
        raw_tensor = torch.tensor(raw_values[group], dtype=torch.float64)
        summaries[group] = {
            "count": int(norm_tensor.numel()),
            "mean": float(raw_tensor.mean().item()),
            "std": float(raw_tensor.std(unbiased=False).item()),
            "normalized_mean": float(norm_tensor.mean().item()),
            "normalized_std": float(norm_tensor.std(unbiased=False).item()),
        }
    return summaries


def flatten_stable_rank_metrics(record: dict) -> dict[str, float]:
    metrics = {"iter": record["iter"]}
    for group, values in record["groups"].items():
        metrics[f"stable_rank/{group}/mean"] = values["mean"]
        metrics[f"stable_rank/{group}/std"] = values["std"]
        metrics[f"stable_rank_norm/{group}/mean"] = values["normalized_mean"]
        metrics[f"stable_rank_norm/{group}/std"] = values["normalized_std"]
    return metrics


def append_stable_rank_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
