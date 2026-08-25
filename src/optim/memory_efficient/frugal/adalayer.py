import math
import warnings
from typing import Callable, Iterable, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
# from torch.optim.optimizer import _dispatch_sqrt

from transformers.utils.versions import require_version

from .proj_optimizer_templates import GaloreOptimizer, CoordOptimizer, BlockOptimizer

class Adalayer(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of Adalayer is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.Adalayer instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    def _is_state_empty(self, state):
        return any(key not in state for key in ["step", "exp_avg", "exp_avg_sq"])

    @torch.no_grad()
    def _init_state(self, example=None, state=None):
        assert isinstance(state, Dict) or state is None
        assert isinstance(example, torch.Tensor) or example is None
        assert not (state is None and example is None), "One of the arguments `state` and `example` should be specified."
        if state is not None and not self._is_state_empty(state):
            state["step"] = 0
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
        else:
            if state is None:
                state = {}
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(example)
            # adalayer
            state["exp_avg_sq"] = torch.tensor(0.0).to(example)
        return state

    @torch.no_grad()
    def _compute_update(self, grad, state, lr, betas, eps, correct_bias, reduce_dim=None, **kwargs):
        state["step"] += 1

        if kwargs.get("is_lm_head", False) and reduce_dim is None:
            reduce_dim = 0 if grad.shape[0] < grad.shape[1] else 1

        if reduce_dim is not None and state["exp_avg_sq"].ndim == 0:
            state["exp_avg_sq"] = torch.zeros((grad.shape[0] if reduce_dim == 1 else grad.shape[1],)).to(grad)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = betas

        exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
        if reduce_dim is None:
            if kwargs.get("sqrt_numel", True):
                exp_avg_sq.mul_(beta2).add_(grad.square().sum().div_(math.sqrt(grad.numel())), alpha=1.0 - beta2)
            else:
                exp_avg_sq.mul_(beta2).add_(grad.square().sum().div_(grad.numel()), alpha=1.0 - beta2)
        else:
            if kwargs.get("sqrt_numel", True):
                exp_avg_sq.mul_(beta2).add_(grad.square().sum(dim=reduce_dim).div_(math.sqrt(grad.shape[reduce_dim])), alpha=1.0 - beta2)
            else:
                exp_avg_sq.mul_(beta2).add_(grad.square().sum(dim=reduce_dim).div_(grad.shape[reduce_dim]), alpha=1.0 - beta2)
        denom = exp_avg_sq.sqrt()

        if reduce_dim is not None:
            denom = denom.unsqueeze(-1 if reduce_dim == 1 else -2)

        step_size = lr
        if correct_bias:
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            denom.div_(bias_correction2_sqrt)
        
        denom.add_(eps)
        
        return exp_avg / denom * (-step_size)


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adalayer does not support sparse gradients")
            
                state = self.state[p]

                if len(state) == 0:
                    self._init_state(example=p, state=state)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                update = self._compute_update(grad, state, group["lr"], group["betas"], group["eps"], group["correct_bias"], 
                                              is_lm_head=group.get("is_lm_head", False), sqrt_numel=group.get("sqrt_numel", True))

                p.add_(update)
        
        return loss


class GaloreAdalayer(GaloreOptimizer, Adalayer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,

        # proj specific
        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # galore specific
        proj_side='std',
        proj_type='svd',

        # adalayer specific
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            proj_side=proj_side,
            proj_type=proj_type,
        )
        Adalayer.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            correct_bias=correct_bias, 
            no_deprecation_warning=no_deprecation_warning
        )


class CoordAdalayer(CoordOptimizer, Adalayer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,

        # proj specific
        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # coord specific
        coord_choice='columns',

        # adalayer specific
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            coord_choice=coord_choice,
        )
        Adalayer.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            correct_bias=correct_bias, 
            no_deprecation_warning=no_deprecation_warning
        )


class BlockAdalayer(BlockOptimizer, Adalayer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        proj_params=None,

        # proj specific
        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # block specific
        block_order='random',

        # adalayer specific
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            block_order=block_order,
        )
        Adalayer.__init__(
            self, params, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            correct_bias=correct_bias, 
            no_deprecation_warning=no_deprecation_warning
        )