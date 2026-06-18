import math
from typing import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer


class LoRAOptimizer(Optimizer):
    """
    LoRA-based optimizer wrapper.

    Non-targeted parameters are passed directly to the base optimizer.

    Args:
        params: Parameter groups (list of dicts).  Groups with
            ``is_proj_params=True`` are LoRA-targeted.
        optimizer_cls: Base optimizer class
            (e.g. ``torch.optim.AdamW``).
        rank: Default LoRA rank *r*.  Can be overridden per-group via
            the ``rank`` key.
        lora_alpha: Scaling factor.  Effective scale is
            ``lora_alpha / rank``.
        **optimizer_kwargs: Forwarded to ``optimizer_cls``.
    """

    def __init__(
        self,
        params,
        optimizer_cls,
        rank: int = 16,
        lora_alpha: float = 1.0,
        **optimizer_kwargs,
    ):
        # Skip Optimizer.__init__ — it rejects empty param lists on newer
        # PyTorch.  We manage param_groups via the internal _base_optimizer.
        torch._C._log_api_usage_once(f"python.optimizer.{self.__class__.__name__}")
        self.defaults = {}
        self.state = {}
        self.param_groups = []

        self._rank = rank
        self._lora_alpha = lora_alpha

        if isinstance(params, torch.Tensor):
            raise TypeError(
                "params must be an iterable of parameter groups (dicts)"
            )
        param_groups = list(params)

        self._lora_factors = {}          # id(p) -> (A, B)
        self._lora_original_params = []  # ordered list of original LoRA-targeted params
        self._lora_group_idx = {}        # id(p) -> index into base-optimizer param_groups
        base_opt_groups = []

        for group in param_groups:
            if not isinstance(group, dict):
                group = {"params": [group]}

            is_lora = group.get("is_proj_params", False)
            group_rank = group.get("rank", rank)

            fwd = {
                k: v
                for k, v in group.items()
                if k
                not in (
                    "params",
                    "is_proj_params",
                    "rank",
                    "update_proj_gap",
                    "proj_type",
                    "proj_side",
                )
            }

            lora_factors = []
            regular = []

            for p in group["params"]:
                if is_lora and p.ndim == 2:
                    m, n = p.shape
                    r = group_rank
                    A = nn.Parameter(
                        torch.randn(r, n, device=p.device, dtype=p.dtype)
                        / math.sqrt(r)
                    )
                    B = nn.Parameter(
                        torch.zeros(m, r, device=p.device, dtype=p.dtype)
                    )
                    self._lora_factors[id(p)] = (A, B)
                    self._lora_original_params.append(p)
                    # Record group index (will be set after groups are appended)
                    self._lora_group_idx[id(p)] = len(base_opt_groups)
                    lora_factors.extend([A, B])
                else:
                    regular.append(p)

            if lora_factors:
                base_opt_groups.append({"params": lora_factors, **fwd})
            if regular:
                base_opt_groups.append({"params": regular, **fwd})

        # Create the base optimizer
        self._base_optimizer = optimizer_cls(base_opt_groups, **optimizer_kwargs)

        # Expose param_groups for scheduler / training-loop compatibility
        self.param_groups = self._base_optimizer.param_groups

        n_lora = len(self._lora_original_params)
        n_total = sum(
            len(g["params"]) for g in base_opt_groups
        )
        print(
            f"LoRAOptimizer: {n_lora} LoRA-targeted weights, "
            f"{n_total} total base-optimizer params, "
            f"rank={rank}, alpha={lora_alpha}, "
            f"base={optimizer_cls.__name__}"
        )

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        scale = self._lora_alpha / self._rank

        saved = {}
        for p in self._lora_original_params:
            if p.grad is None:
                continue
            A, B = self._lora_factors[id(p)]
            G = p.grad

            # Gradients for the LoRA factors
            A.grad = B.data.T @ G       # (r, n)
            B.grad = G @ A.data.T       # (m, r)

            saved[id(p)] = (A.data.clone(), B.data.clone())

        self._base_optimizer.step()

        # 3. Apply LoRA deltas to the original weights.
        for p in self._lora_original_params:
            pid = id(p)
            if pid not in saved:
                continue
            A, B = self._lora_factors[pid]
            A_old, B_old = saved[pid]

            # Weight decay on the original weight
            grp = self._base_optimizer.param_groups[self._lora_group_idx[pid]]
            wd = grp.get("weight_decay", 0.0)
            if wd > 0:
                p.data.mul_(1.0 - grp["lr"] * wd)

            delta_A = A.data - A_old     # (r, n)
            delta_B = B.data - B_old     # (m, r)

            p.data.addmm_(delta_B, A.data, alpha=scale)   # += scale * dB @ A_new
            p.data.addmm_(B_old, delta_A, alpha=scale)    # += scale * B_old @ dA

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for LoRA factors *and* original LoRA-targeted params."""
        self._base_optimizer.zero_grad(set_to_none=set_to_none)
        for p in self._lora_original_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_dict(self):
        lora_factors = {}
        for i, p in enumerate(self._lora_original_params):
            A, B = self._lora_factors[id(p)]
            lora_factors[i] = {"A": A.data, "B": B.data}
        return {
            "base_optimizer": self._base_optimizer.state_dict(),
            "lora_factors": lora_factors,
            "rank": self._rank,
            "lora_alpha": self._lora_alpha,
        }

    def load_state_dict(self, state_dict):
        self._base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self._base_optimizer.param_groups
        for i, p in enumerate(self._lora_original_params):
            A, B = self._lora_factors[id(p)]
            factors = state_dict["lora_factors"][i]
            A.data.copy_(factors["A"])
            B.data.copy_(factors["B"])
