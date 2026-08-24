"""Per-layer stable-rank and spectrum logging based on ``ruslan_logs``.

Tracking protocol:
  - Track every transformer block.
  - Track Q/K/V/O and the MLP up/gate/down projections separately.
  - Log stable rank for every individual matrix. Nothing is averaged across
    matrices, blocks, or iterations.
  - Log one accumulated log10 singular-value line plot for each
    (layer, projection). Each snapshot is a separate ``step_N`` curve.

Normalized stable rank is
``(||W||_F^2 / ||W||_2^2) / min(rows, columns)`` and lies in (0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import wandb

from models.tucker_linear import BlockTermTuckerLinear, TuckerLinear


_EPS = 1e-12
_MAX_EFFECTIVE_WEIGHT_ELEMENTS = 50_000_000
_SPECTRUM_HISTORY: dict[str, list[tuple[int, torch.Tensor]]] = {}

_PROJECTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("attn", "q_proj"), "q_proj"),
    (("attn", "k_proj"), "k_proj"),
    (("attn", "v_proj"), "v_proj"),
    (("attn", "o_proj"), "o_proj"),
    (("mlp", "up_proj"), "up_proj"),
    (("mlp", "gate_proj"), "gate_proj"),
    (("mlp", "down_proj"), "down_proj"),
)


def is_spectral_snapshot_step(curr_iter: int, interval: int) -> bool:
    """Return whether this iteration is exactly on the requested cadence."""
    return interval > 0 and curr_iter > 0 and curr_iter % interval == 0


def reset_spectrum_history() -> None:
    """Clear accumulated W&B spectrum curves for a new process-local run."""
    _SPECTRUM_HISTORY.clear()


def _accumulated_spectrum_plot(
    key: str,
    curr_iter: int,
    spectrum: torch.Tensor,
):
    """Return one W&B chart containing every spectrum snapshot seen so far."""
    history = _SPECTRUM_HISTORY.setdefault(key, [])
    snapshot = spectrum.detach().cpu()
    if history and history[-1][0] == curr_iter:
        history[-1] = (curr_iter, snapshot)
    else:
        history.append((curr_iter, snapshot))

    return wandb.plot.line_series(
        xs=[list(range(values.numel())) for _, values in history],
        ys=[values.tolist() for _, values in history],
        keys=[f"step_{step}" for step, _ in history],
        title=key,
        xname="singular value index",
    )


def _tracked_block_indices(n_layer: int) -> list[int]:
    """Return every transformer block index."""
    return list(range(n_layer))


def _resolve_attr(module: nn.Module, path: tuple[str, ...]) -> nn.Module | None:
    for attribute in path:
        module = getattr(module, attribute, None)
        if module is None:
            return None
    return module


@torch.no_grad()
def _effective_weight(
    module: nn.Module,
) -> torch.Tensor | None:
    """Return the logical matrix used by a dense or Tucker module.

    Stable rank is deliberately computed on the reconstructed effective
    matrix, never averaged over Tucker factors or the core.
    """
    if isinstance(module, (TuckerLinear, BlockTermTuckerLinear)):
        try:
            return module.materialize_weight(
                dtype=torch.float32,
                max_elements=_MAX_EFFECTIVE_WEIGHT_ELEMENTS,
            ).detach()
        except ValueError as error:
            print(f"Skipping Tucker spectrum metric: {error}")
            return None
    if isinstance(module, (nn.Linear, nn.Embedding)):
        return module.weight.detach().float()
    return None


@torch.no_grad()
def _normalized_stable_rank_and_sv(
    weight: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    singular_values = torch.linalg.svdvals(weight)
    frobenius_squared = (singular_values**2).sum()
    spectral_squared = singular_values[0] ** 2 + _EPS
    normalization = singular_values.shape[-1]
    stable_rank = (
        frobenius_squared / spectral_squared / normalization
    ).item()
    return stable_rank, singular_values


@torch.no_grad()
def log_spectrum_and_stable_rank(
    raw_model: torch.nn.Module,
    curr_iter: int,
    log_stable_rank: bool = True,
    log_full_spectrum: bool = False,
) -> None:
    """Log per-layer stable ranks and/or accumulated log10 spectra."""
    blocks = raw_model.transformer.h
    logs: dict[str, object] = {"iter": curr_iter}

    for block_index in _tracked_block_indices(len(blocks)):
        block = blocks[block_index]
        for attribute_path, projection_name in _PROJECTIONS:
            module = _resolve_attr(block, attribute_path)
            if module is None:
                continue
            weight = _effective_weight(module)
            if weight is None:
                continue

            stable_rank, singular_values = _normalized_stable_rank_and_sv(weight)
            key_prefix = f"layer{block_index:02d}_{projection_name}"
            if log_stable_rank:
                logs[f"stable_rank/{key_prefix}"] = stable_rank
            if log_full_spectrum:
                log_singular_values = torch.log10(
                    singular_values.clamp_min(_EPS)
                )
                spectrum_key = f"spectrum/{key_prefix}_log_spectrum"
                plot_title = (
                    f"Layer {block_index} — {projection_name} log spectrum"
                )
                logs[spectrum_key] = _accumulated_spectrum_plot(
                    plot_title,
                    curr_iter,
                    log_singular_values,
                )

    wandb.log(logs)
