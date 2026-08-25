"""
MuonBP — Muon with Block-Periodic orthogonalization.

Reference: "MuonBP: Faster Muon via Block-Periodic Orthogonalization"
(https://arxiv.org/abs/2510.16981).

Idea. Plain Muon orthogonalizes the *full* momentum matrix every step via
Newton-Schulz. In a model-parallel setting that full orthogonalization needs a
gather/scatter of the sharded matrix and costs 5-10% throughput. MuonBP instead
orthogonalizes each matrix *shard/block independently* on most steps ("block
step") and only performs a *full* orthogonalization every ``period`` steps
("full step"), recovering Muon's optimization quality at a fraction of the
communication.

Two stepsizes (the paper's practical rule). The paper matches the Adam-style
RMS norm of the update by scaling with the matrix dimensions: block steps are
scaled by the dimensions of the *smaller* (block) matrix, full steps by the
dimensions of the *full* matrix. That dimension-based scaling is exactly the
``(rows / cols) ** 0.5`` factor below, so the ratio between the effective block
and full learning rate is realized automatically from a single base ``lr``
(= the "full" learning rate). Tying the two stepsizes is what the paper reports
as worse-converging, so we keep the dimension-based scaling on both paths.

This is the single-GPU analogue: instead of device shards we partition each 2-D
projection matrix into ``nblocks`` contiguous blocks along its larger dimension
(``nblocks=2`` mirrors the paper's small-model TP=2 setting).

Structure mirrors optim/monarch_muon.py: one torch.optim.Optimizer with two
param groups (Muon for 2-D projection matrices, AdamW for 1-D params,
embeddings and lm_head) so a standard LR scheduler can drive both via
param_groups.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization (5 iterations, Muon coefficients)
# ---------------------------------------------------------------------------

def _newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz orthogonalization for a single 2-D matrix."""
    m = matrix / (matrix.norm(p="fro") + 1e-7)
    transposed = False
    if m.shape[0] > m.shape[1]:
        m = m.T
        transposed = True
    for _ in range(5):
        mm_t = m @ m.T
        m = 3.4445 * m - 4.7750 * mm_t @ m + 2.0315 * (mm_t @ mm_t) @ m
    if transposed:
        m = m.T
    return m


# ---------------------------------------------------------------------------
# Combined optimizer
# ---------------------------------------------------------------------------

class MuonBPOptimizer(torch.optim.Optimizer):
    """MuonBP (block-periodic Muon) + AdamW in a single Optimizer object.

    Parameters
    ----------
    muon_params : list[Tensor]
        Parameters updated with the (block-periodic) Muon rule — 2-D projection
        matrices, excluding embeddings and lm_head.
    adamw_params : list[Tensor]
        Parameters updated with AdamW (1-D params, embeddings, lm_head).
    lr : float
        Base ("full") learning rate, shared by both groups (the scheduler
        scales it). The effective block-step learning rate follows from the
        dimension-based RMS scaling.
    momentum : float
        Momentum for the Muon heavy-ball buffer.
    nblocks : int
        Number of contiguous blocks each Muon matrix is partitioned into on a
        block step (paper: number of matrix shards, = TP degree).
    period : int
        Do a *full* orthogonalization every ``period`` steps; block-orthogonalize
        on the others. ``period=5`` is the paper's recommended value; ``period<=1``
        reduces to plain Muon (full every step).
    adamw_betas, adamw_weight_decay, adamw_eps :
        Standard AdamW hyperparameters for the AdamW group.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nblocks: int = 2,
        period: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_weight_decay: float = 0.1,
        adamw_eps: float = 1e-8,
    ):
        if nblocks < 1:
            raise ValueError(f"nblocks must be >= 1, got {nblocks}")
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)

        self.nblocks = int(nblocks)
        self.period = int(period)
        self._step_count = 0  # global optimizer step, drives the block/full phase

        defaults = dict(lr=lr)
        param_groups = [
            {
                "params": muon_params,
                "update_type": "muon",
                "momentum": momentum,
            },
            {
                "params": adamw_params,
                "update_type": "adamw",
                "betas": adamw_betas,
                "weight_decay": adamw_weight_decay,
                "eps": adamw_eps,
            },
        ]
        super().__init__(param_groups, defaults)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        # Full orthogonalization every `period`-th step, block otherwise.
        full_step = (self.period <= 1) or (self._step_count % self.period == 0)

        for group in self.param_groups:
            lr = group["lr"]
            if group["update_type"] == "muon":
                self._muon_step(group, lr, full_step)
            else:
                self._adamw_step(group, lr)

        return loss

    # ------------------------------------------------------------------
    # Private: Muon update (block-periodic)
    # ------------------------------------------------------------------

    def _get_momentum_buffer(self, param: torch.Tensor) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)
        return state["momentum_buffer"]

    def _muon_step(self, group: dict, lr: float, full_step: bool) -> None:
        momentum = group["momentum"]

        for param in group["params"]:
            if param.grad is None:
                continue
            buf = self._get_momentum_buffer(param)
            buf.mul_(momentum).add_(param.grad)

            # Flatten anything >2-D to 2-D (llama projections are already 2-D).
            mat = buf.reshape(buf.shape[0], -1) if buf.ndim > 2 else buf
            pdata = param.data.reshape(mat.shape) if param.data.shape != mat.shape else param.data

            if full_step or self.nblocks == 1:
                # Full orthogonalization; RMS-scale by the full matrix dims.
                ortho = _newton_schulz(mat)
                scale = (mat.shape[0] / mat.shape[1]) ** 0.5
                pdata.add_(ortho, alpha=-lr * scale)
            else:
                # Block-orthogonalize independent shards along the larger dim;
                # RMS-scale by each block's own dims.
                split_dim = 0 if mat.shape[0] >= mat.shape[1] else 1
                buf_blocks = torch.tensor_split(mat, self.nblocks, dim=split_dim)
                param_blocks = torch.tensor_split(pdata, self.nblocks, dim=split_dim)
                for pblk, bblk in zip(param_blocks, buf_blocks):
                    if bblk.numel() == 0:
                        continue
                    ortho = _newton_schulz(bblk)
                    scale = (bblk.shape[0] / bblk.shape[1]) ** 0.5
                    pblk.add_(ortho, alpha=-lr * scale)

    # ------------------------------------------------------------------
    # Private: AdamW update
    # ------------------------------------------------------------------

    def _adamw_step(self, group: dict, lr: float) -> None:
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            grad = param.grad

            if weight_decay > 0:
                param.data.mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = lr / bc1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            param.data.addcdiv_(exp_avg, denom, value=-step_size)
