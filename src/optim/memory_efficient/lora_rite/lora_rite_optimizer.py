import math

import torch
from torch import nn
from torch.optim import Optimizer

from .lora_rite import LoRARite


class LoRARiteOptimizer(Optimizer):
    """
    LoRA-Rite optimizer wrapper.

    Creates low-rank factors (A, B) for parameters marked with
    ``is_proj_params=True`` and optimises them using the LoRA-Rite
    algorithm.  Non-targeted parameters are passed to a standard
    base optimizer (AdamW by default).

    Args:
        params: Parameter groups (list of dicts).  Groups with
            ``is_proj_params=True`` are LoRA-targeted.
        rank: Default LoRA rank *r*.  Can be overridden per-group.
        lora_alpha: Scaling factor (effective scale = alpha / rank).
        lr: Learning rate.
        betas: Momentum coefficients for LoRA-Rite and AdamW.
        eps: Epsilon for numerical stability.
        weight_decay: Weight decay coefficient.
        base_optimizer_cls: Optimizer class for non-LoRA parameters
            (default: ``torch.optim.AdamW``).
        clip_unmagnified_grad: LoRA-Rite gradient clipping threshold.
        update_capping: LoRA-Rite update capping threshold (0 = off).
        update_skipping: LoRA-Rite update skipping threshold.
        apply_escape: Enable escape mechanism in LoRA-Rite.
        balance_param: Balance LoRA factor norms after each step.
    """

    def __init__(
        self,
        params,
        rank: int = 16,
        lora_alpha: float = 1.0,
        lr: float = 1e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        base_optimizer_cls=None,
        clip_unmagnified_grad: float = 1.0,
        update_capping: float = 0.0,
        update_skipping: float = 1.0,
        apply_escape: bool = False,
        balance_param: bool = False,
    ):
        # We manage param_groups via sub-optimizers, skip Optimizer.__init__
        torch._C._log_api_usage_once(f"python.optimizer.{self.__class__.__name__}")
        self.defaults = {}
        self.state = {}
        self.param_groups = []

        self._rank = rank
        self._lora_alpha = lora_alpha

        if isinstance(params, torch.Tensor):
            raise TypeError("params must be an iterable of parameter groups (dicts)")
        param_groups = list(params)

        self._lora_factors = {}           # id(p) -> (A, B)
        self._lora_original_params = []   # ordered list of original LoRA-targeted params
        lora_factor_list = []             # interleaved [A1, B1, A2, B2, ...]
        non_lora_groups = []

        for group in param_groups:
            if not isinstance(group, dict):
                group = {"params": [group]}

            is_lora = group.get("is_proj_params", False)
            group_rank = group.get("rank", rank)

            # Forward keys relevant to base optimizer (lr, weight_decay, etc.)
            fwd = {
                k: v
                for k, v in group.items()
                if k not in (
                    "params", "is_proj_params", "rank",
                    "update_proj_gap", "proj_type", "proj_side",
                )
            }

            lora_factors = []
            regular = []

            for p in group["params"]:
                if is_lora and p.ndim == 2:
                    m, n = p.shape
                    r = group_rank
                    A = nn.Parameter(
                        torch.randn(r, n, device=p.device, dtype=p.dtype) / math.sqrt(r)
                    )
                    B = nn.Parameter(
                        torch.zeros(m, r, device=p.device, dtype=p.dtype)
                    )
                    self._lora_factors[id(p)] = (A, B)
                    self._lora_original_params.append(p)
                    # LoRA-Rite expects pairs [A, B] with lora_l_dim=0, lora_r_dim=-1
                    lora_factors.extend([A, B])
                else:
                    regular.append(p)

            if lora_factors:
                lora_factor_list.extend(lora_factors)
            if regular:
                non_lora_groups.append({"params": regular, **fwd})

        # --- LoRA-Rite sub-optimizer for factor pairs ---
        self._lora_rite = None
        if lora_factor_list:
            lora_group = {
                "params": lora_factor_list,
                "lr": lr,
                "weight_decay": weight_decay,
            }
            self._lora_rite = LoRARite(
                [lora_group],
                betas=betas,
                eps=eps,
                lr=lr,
                weight_decay=weight_decay,
                clip_unmagnified_grad=clip_unmagnified_grad,
                update_capping=update_capping,
                update_skipping=update_skipping,
                apply_escape=apply_escape,
                balance_param=balance_param,
                lora_l_dim=0,
                lora_r_dim=-1,
            )

        # --- Base optimizer for non-LoRA parameters ---
        base_cls = base_optimizer_cls or torch.optim.AdamW
        self._base_optimizer = None
        if non_lora_groups:
            base_kwargs = dict(lr=lr, weight_decay=weight_decay)
            if base_cls in (torch.optim.AdamW, torch.optim.Adam):
                base_kwargs.update(betas=betas, eps=eps)
            self._base_optimizer = base_cls(non_lora_groups, **base_kwargs)

        # Expose combined param_groups for LR scheduler / training loop
        all_groups = []
        if self._lora_rite:
            all_groups.extend(self._lora_rite.param_groups)
        if self._base_optimizer:
            all_groups.extend(self._base_optimizer.param_groups)
        self.param_groups = all_groups

        n_lora = len(self._lora_original_params)
        n_factors = len(lora_factor_list)
        n_regular = sum(len(g["params"]) for g in non_lora_groups) if non_lora_groups else 0
        print(
            f"LoRARiteOptimizer: {n_lora} LoRA-targeted weights, "
            f"{n_factors} LoRA factors, {n_regular} regular params, "
            f"rank={rank}, alpha={lora_alpha}, base={base_cls.__name__}"
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        scale = self._lora_alpha / self._rank

        # 1. Compute gradients for LoRA factors from original weight gradients
        saved = {}
        for p in self._lora_original_params:
            A, B = self._lora_factors[id(p)]
            if p.grad is None:
                # Set zero gradients so LoRA-Rite doesn't crash on missing grads
                A.grad = torch.zeros_like(A)
                B.grad = torch.zeros_like(B)
                continue
            G = p.grad
            A.grad = B.data.T @ G       # (r, n)
            B.grad = G @ A.data.T       # (m, r)
            saved[id(p)] = (A.data.clone(), B.data.clone())

        # 2. Run LoRA-Rite step on factor pairs
        if self._lora_rite:
            self._lora_rite.step()

        # 3. Run base optimizer step on non-LoRA params
        if self._base_optimizer:
            self._base_optimizer.step()

        # 4. Apply LoRA deltas to original weights
        for p in self._lora_original_params:
            pid = id(p)
            if pid not in saved:
                continue
            A, B = self._lora_factors[pid]
            A_old, B_old = saved[pid]

            delta_A = A.data - A_old     # (r, n)
            delta_B = B.data - B_old     # (m, r)

            p.data.addmm_(delta_B, A.data, alpha=scale)
            p.data.addmm_(B_old, delta_A, alpha=scale)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for LoRA factors, original LoRA-targeted params, and base params."""
        if self._lora_rite:
            self._lora_rite.zero_grad(set_to_none=set_to_none)
        if self._base_optimizer:
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

        # Manually save LoRA-Rite internal state (SimpleNamespace not always portable)
        lora_rite_state = None
        if self._lora_rite:
            rite = self._lora_rite
            pair_states = {}
            for group in rite.param_groups:
                pairs = list(zip(group["params"], group["params"][1:]))[::2]
                for idx, (p1, _p2) in enumerate(pairs):
                    if p1 in rite.state and "attr" in rite.state[p1]:
                        s = rite.state[p1]["attr"]
                        pair_states[idx] = {
                            "step": s.step,
                            "v_l": s.v_l, "v_r": s.v_r,
                            "m_l": s.m_l, "m_r": s.m_r,
                            "basis_l": s.basis_l, "basis_r": s.basis_r,
                            "escape_l": s.escape_l, "escape_r": s.escape_r,
                        }
            lora_rite_state = {
                "count": rite.count,
                "state_initialized": rite.state_initialized,
                "pair_states": pair_states,
            }

        return {
            "lora_rite_state": lora_rite_state,
            "base_optimizer": self._base_optimizer.state_dict() if self._base_optimizer else None,
            "lora_factors": lora_factors,
            "rank": self._rank,
            "lora_alpha": self._lora_alpha,
        }

    def load_state_dict(self, state_dict):
        # Restore LoRA factors
        for i, p in enumerate(self._lora_original_params):
            A, B = self._lora_factors[id(p)]
            factors = state_dict["lora_factors"][i]
            A.data.copy_(factors["A"])
            B.data.copy_(factors["B"])

        # Restore base optimizer
        if self._base_optimizer and state_dict["base_optimizer"]:
            self._base_optimizer.load_state_dict(state_dict["base_optimizer"])

        # Restore LoRA-Rite internal state
        if self._lora_rite and state_dict["lora_rite_state"]:
            import types
            rite = self._lora_rite
            saved = state_dict["lora_rite_state"]
            rite.count = saved["count"]
            rite.state_initialized = saved["state_initialized"]

            for group in rite.param_groups:
                pairs = list(zip(group["params"], group["params"][1:]))[::2]
                for idx, (p1, _p2) in enumerate(pairs):
                    if idx in saved["pair_states"]:
                        ps = saved["pair_states"][idx]
                        s = types.SimpleNamespace()
                        s.step = ps["step"]
                        s.v_l = ps["v_l"]
                        s.v_r = ps["v_r"]
                        s.m_l = ps["m_l"]
                        s.m_r = ps["m_r"]
                        s.basis_l = ps["basis_l"]
                        s.basis_r = ps["basis_r"]
                        s.escape_l = ps["escape_l"]
                        s.escape_r = ps["escape_r"]
                        rite.state[p1]["attr"] = s

        # Update combined param_groups reference
        all_groups = []
        if self._lora_rite:
            all_groups.extend(self._lora_rite.param_groups)
        if self._base_optimizer:
            all_groups.extend(self._base_optimizer.param_groups)
        self.param_groups = all_groups
