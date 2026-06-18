# Copyright (C) 2025 ByteDance
# Modifications by project authors: density-based rank, standalone integration.
# Licensed under the Apache License, Version 2.0

import math
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from .coap_matrix import Projector, MatrixCOAP


class COAPAdamW(Optimizer):
    """
    AdamW with COAP gradient projection.

    COAP projects 2-D weight gradients into a low-rank subspace using an
    adaptively-updated orthogonal basis (cheap gradient-descent update every
    ``update_interval`` steps; full SVD recompute every
    ``update_interval * reproject_factor`` steps).

    Args:
        params: model parameters or param-group list.
        lr: learning rate.
        betas: Adam (beta1, beta2).
        eps: Adam epsilon.
        weight_decay: decoupled weight decay.
        correct_bias: whether to apply bias correction.
        density: fraction of the smaller dimension used as rank,
            e.g. 0.25 → rank = max(1, int(0.25 * min(p.shape))).
        update_interval: steps between cheap projection updates.
        reproject_factor: full SVD recomputed every
            update_interval * reproject_factor steps.
        restore_state: rotate Adam moments into the new basis on update.
        scale: scalar multiplier applied when projecting back.
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            density: float = 0.25,
            update_interval: int = 32,
            reproject_factor: int = 5,
            restore_state: bool = False,
            scale: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 < density <= 1.0:
            raise ValueError(f"Invalid density: {density} — must be in (0, 1]")

        self.density = density
        self.update_interval = update_interval
        self.reproject_factor = reproject_factor
        self.restore_state = restore_state
        self.scale = scale

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("COAPAdamW does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                # Initialise projector on first step
                if "projector" not in state:
                    if grad.ndim == 2:
                        rank = max(1, int(self.density * min(grad.shape)))
                        state["projector"] = MatrixCOAP(
                            rank=rank,
                            update_interval=self.update_interval,
                            reproject_factor=self.reproject_factor,
                            restore_state=self.restore_state,
                            scale=self.scale,
                        )
                    else:
                        # 1-D, embeddings, norms — no projection
                        state["projector"] = Projector()

                grad = state["projector"].project(grad, state)

                # State initialisation (in projected space)
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                norm_grad = exp_avg / denom
                # Project back to full-rank space
                norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
