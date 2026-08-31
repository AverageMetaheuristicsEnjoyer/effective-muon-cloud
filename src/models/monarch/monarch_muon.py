"""
MonarchMuon optimizer — Muon with batched Newton-Schulz for 3-D block-diagonal
parameters (blkdiag1 / blkdiag2 from MonarchLinear) plus an embedded AdamW for
embeddings, biases, layer-norm weights, and the lm_head.

Design: a single torch.optim.Optimizer with two param groups so that a
standard LR scheduler can wrap it and update both groups via param_groups.

  Group 0 (update_type="muon"):  2-D and 3-D projection matrices
  Group 1 (update_type="adamw"): 1-D params, embeddings, lm_head

Usage (from main.py):
    from optim.monarch_muon import MonarchMuonOptimizer
    opt = MonarchMuonOptimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        muon_weight_decay=args.weight_decay,
        adamw_betas=(args.beta1, args.beta2),
        adamw_weight_decay=args.weight_decay,
        adamw_eps=args.eps,
    )
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


# ---------------------------------------------------------------------------
# Newton-Schulz helpers
# ---------------------------------------------------------------------------

def _newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz orthogonalisation for a single 2-D matrix."""
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


def _newton_schulz_batched(B: torch.Tensor) -> torch.Tensor:
    """Batched Newton-Schulz for a stack of matrices: (nblocks, d1, d2)."""
    norms = B.norm(p="fro", dim=(-2, -1), keepdim=True)
    B = B / (norms + 1e-7)
    if B.shape[-2] > B.shape[-1]:
        B = B.transpose(-2, -1)
        do_T = True
    else:
        do_T = False
    for _ in range(5):
        BBt = B @ B.transpose(-2, -1)
        B = 3.4445 * B - 4.7750 * BBt @ B + 2.0315 * (BBt @ BBt) @ B
    if do_T:
        B = B.transpose(-2, -1)
    return B


# ---------------------------------------------------------------------------
# Combined optimizer
# ---------------------------------------------------------------------------

class MonarchMuonOptimizer(torch.optim.Optimizer):
    """Muon (with 3-D batching) + AdamW in a single Optimizer object.

    Parameters
    ----------
    muon_params : list[Tensor]
        Parameters to update with the Muon rule (ndim >= 2, not embeddings).
    adamw_params : list[Tensor]
        Parameters to update with AdamW (1-D, embeddings, lm_head).
    lr : float
        Base learning rate, shared by both groups (scheduler will scale it).
    momentum : float
        Momentum for the Muon group.
    nesterov : bool
        Use the same Nesterov momentum rule as the dense Muon optimizer.
    muon_weight_decay : float
        Decoupled weight decay for the Monarch factor group.
    adamw_betas : tuple[float, float]
        (beta1, beta2) for the AdamW group.
    adamw_weight_decay : float
        Weight decay for the AdamW group.
    adamw_eps : float
        Epsilon for the AdamW group.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        *,
        lr: float = 5e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        muon_weight_decay: float = 0.1,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_weight_decay: float = 0.1,
        adamw_eps: float = 1e-8,
        ns_dtype: torch.dtype = torch.float32,
        use_foreach: bool = False,
    ):
        self.ns_dtype = ns_dtype
        self.use_foreach = use_foreach
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)

        defaults = dict(lr=lr)
        param_groups = [
            {
                "params": muon_params,
                "update_type": "muon",
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": muon_weight_decay,
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

        for group in self.param_groups:
            lr = group["lr"]
            if group["update_type"] == "muon":
                self._muon_step(group, lr)
            else:
                self._adamw_step(group, lr)

        return loss

    # ------------------------------------------------------------------
    # Private: Muon update (handles 2-D and 3-D params)
    # ------------------------------------------------------------------

    def _get_momentum_buffer(self, param: torch.Tensor) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)
        return state["momentum_buffer"]

    def _muon_step(self, group: dict, lr: float) -> None:
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]

        params = [p for p in group["params"] if p.grad is not None]
        bufs = [self._get_momentum_buffer(p) for p in params]
        grads = [p.grad for p in params]

        # momentum update: buf = momentum * buf + grad
        if self.use_foreach and bufs:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)
        else:
            for buf, grad in zip(bufs, grads):
                buf.mul_(momentum).add_(grad)

        if weight_decay > 0:
            for param in params:
                param.data.mul_(1.0 - lr * weight_decay)

        updates = [buf.mul(momentum).add(grad) for buf, grad in zip(bufs, grads)] \
            if group["nesterov"] else bufs

        # Accumulate 3-D params by (device, dtype, a, b) for batched NS
        grouped_3d: dict = defaultdict(list)
        for param, update in zip(params, updates):
            if update.ndim == 3:
                key = (update.device, update.dtype, update.shape[1], update.shape[2])
                grouped_3d[key].append((param, update))
            else:
                # 2-D (or higher flattened to 2-D) — regular NS
                reshaped = update.reshape(update.shape[0], -1) if update.ndim > 2 else update
                ortho = _newton_schulz(
                    reshaped.to(self.ns_dtype)).to(update.dtype).reshape_as(update)
                scale = (reshaped.shape[0] / reshaped.shape[1]) ** 0.5
                param.data.add_(ortho, alpha=-lr * scale)

        # Batched NS for same-shaped 3-D blocks
        for (_, _, a, b), items in grouped_3d.items():
            merged = torch.cat([buf for _, buf in items], dim=0)
            if merged.dtype != self.ns_dtype:
                merged = merged.to(self.ns_dtype)
            ortho_merged = _newton_schulz_batched(merged)
            if ortho_merged.dtype != items[0][1].dtype:
                ortho_merged = ortho_merged.to(items[0][1].dtype)
            scale = (a / b) ** 0.5
            if self.use_foreach:
                updates = list(ortho_merged.split([buf.shape[0] for _, buf in items], dim=0))
                torch._foreach_add_([p.data for p, _ in items], updates,
                                    alpha=-lr * scale)
            else:
                offset = 0
                for param, buf in items:
                    n = buf.shape[0]
                    param.data.add_(ortho_merged[offset:offset + n], alpha=-lr * scale)
                    offset += n

    # ------------------------------------------------------------------
    # Private: AdamW update
    # ------------------------------------------------------------------

    def _adamw_step(self, group: dict, lr: float) -> None:
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        if self.use_foreach:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                return
            for param in params:
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
            steps = [self.state[p]["step"] for p in params]
            if len(set(steps)) == 1:
                step = steps[0]
                grads = [p.grad for p in params]
                exp_avgs = [self.state[p]["exp_avg"] for p in params]
                exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in params]
                if weight_decay > 0:
                    torch._foreach_mul_([p.data for p in params], 1.0 - lr * weight_decay)
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_div_(denom, math.sqrt(bc2))
                torch._foreach_add_(denom, eps)
                torch._foreach_addcdiv_([p.data for p in params], exp_avgs, denom,
                                        value=-lr / bc1)
                return
            # non-uniform steps: fall through to the per-param path below
            # (steps were already incremented, so decrement to avoid double count)
            for param in params:
                self.state[param]["step"] -= 1

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
