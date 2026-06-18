# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
# from .galore_projector_tensor import GaLoreProjectorTensor


class AdaMeM(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        adamem_relative_lr: float = 1.0,
        use_momentum_to_update_variance: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
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
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                    "adamem_relative_lr": adamem_relative_lr, 
                    "use_momentum_to_update_variance": use_momentum_to_update_variance}
        super().__init__(params, defaults)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).sqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).sqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _approx_sq_grad_onesided(exp_avg_sq, grad_shape): #, adamem_type="rowwise"):
        # if adamem_type == "rowwise":
        r_factor = exp_avg_sq.sqrt().unsqueeze(-1)
        c_factor = torch.full(size=(grad_shape[1],), fill_value=1. / grad_shape[1], dtype=r_factor.dtype, device=r_factor.device).sqrt_().unsqueeze(-2)
        # else:
        #     c_factor = exp_avg_sq.sqrt().unsqueeze(-2)
        #     r_factor = torch.full(size=(grad_shape[0],), fill_value=1. / grad_shape[0], dtype=c_factor.dtype, device=c_factor.device).sqrt_().unsqueeze(-1)
        return torch.mul(r_factor, c_factor)

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
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                    
                is_proj_group = "rank" in group

                # GaLore Projection
                if is_proj_group:
                    if "projector" not in state:
                        if group['dim'] <=2:
                            state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            raise ValueError("not supporting >2 dim tensor for now")
                            # state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)

                    # Exponential moving average of squared gradient values
                    if is_proj_group:
                        state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:]).to(grad)

                        state["os_exp_avg_sq"] = torch.zeros(p.grad.shape[:-1]).to(grad)

                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** state["step"])

                # Decay the first moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))

                # Decay the second moment running average coefficient
                if is_proj_group:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    update_tensor = exp_avg.square() if group["use_momentum_to_update_variance"] else grad.square()

                    exp_avg_sq_row.mul_(beta2).add_(update_tensor.mean(dim=-1), alpha=(1.0 - beta2))
                    exp_avg_sq_col.mul_(beta2).add_(update_tensor.mean(dim=-2), alpha=(1.0 - beta2))

                    # Approximation of exponential moving average of square of gradient
                    denom = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)

                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt()
                
                # bias correction and eps
                denom.div_(bias_correction2_sqrt)
                denom.add_(group["eps"])

                step_size = group["lr"]

                # compute norm gradient
                norm_grad = exp_avg.div(bias_correction1) / denom
                
                # GaLore Projection Back
                if is_proj_group:
                    norm_grad = state["projector"].project_back(norm_grad)

                    # Decay the second moment running average coefficient
                    # for one sided Adafactor preconditioner and
                    # update `norm_grad` with modified residual
                    residual_grad = p.grad - state["projector"].project_back(grad)
                    adamem_update = residual_grad.square()

                    os_exp_avg_sq = state["os_exp_avg_sq"]
                    avg_dim = -1 # if group["adamem_type"] == "rowwise" else -2
                    if group["adamem_reduce_op"] == "mean":
                        os_exp_avg_sq.mul_(beta2).add_(adamem_update.mean(dim=avg_dim), alpha=(1.0 - beta2))
                    else:
                        os_exp_avg_sq.mul_(beta2).add_(adamem_update.sum(dim=avg_dim), alpha=(1.0 - beta2))

                    residual_denom = self._approx_sq_grad_onesided(os_exp_avg_sq, adamem_update.shape) #, group["adamem_type"])
                    
                    # bias correction and eps
                    residual_denom.div_(bias_correction2_sqrt)
                    residual_denom.add_(group["eps"])

                    adamem_update = residual_grad / residual_denom

                    adamem_update.mul_(group["adamem_relative_lr"])

                    norm_grad.add_(adamem_update)

                
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
