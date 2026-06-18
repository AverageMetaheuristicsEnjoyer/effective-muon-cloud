import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, List, Dict, Iterable
from abc import abstractmethod

from .galore_projector import GaLoreProjector
from .coordinate_projector import CoordinateProjector

import warnings

def prepare_for_majority_vote_signsgd(ddp_model):
    vocab_size = None
    for name, p in ddp_model.named_parameters():
        if "embed" in name:
            vocab_size = max(p.size()[0], p.size()[1])
    assert vocab_size is not None

    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    def majority_vote_hook(state, bucket):
        grads_flattened = bucket.buffer()
        grads = bucket.gradients()
        offset = 0
        for grad in grads:
            if grad.ndim == 2 and all(dim < vocab_size for dim in grad.size()):
                grads_flattened.narrow(0, offset, grad.numel()).sign_()
            offset += grad.numel()
        return default_hooks.allreduce_hook(state, bucket)

    ddp_model.register_comm_hook(state=None, hook=majority_vote_hook)


def prepare_proj_params(model, target_modules_list=None, proj_norms=False, proj_embeds=False, proj_logits=False, fsdp=False, use_orig_params=False, lm_head_separate_group=False, qkv_separate_group=False):
    if fsdp and (lm_head_separate_group or qkv_separate_group or proj_norms or proj_embeds or proj_logits):
        warnings.warn("`fsdp` may not work with `proj_{smth}` or `{some}_separate_group`")
    if fsdp and not use_orig_params:
        assert proj_norms, "transformer_auto_wrap_policy wraps entire `layer`, therefore it is impossible to separate Norms and Linear layers."
        assert not (proj_embeds ^ proj_logits), "transformer_auto_wrap_policy wraps embeds and logits to one unit, therefore it is impossible to separate them."
    if target_modules_list is None:
        target_modules_list = ["attn", "mlp", 
                               "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj",
                               "query", "value", "key", "intermediate.dense", "output.dense"]
    if qkv_separate_group:
        qkv_modules_list = [
            "q_proj", "k_proj", "v_proj",
            "query", "key", "value"
        ]
    proj_params = []
    qkv_params = []
    lm_head_params = []
    if fsdp and not use_orig_params:
        for name, p in model.named_parameters():
            if not "layer" in name:
                continue

            proj_params.append(p)
    else:
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            curr_params_list = proj_params
            if qkv_separate_group and any(target_key in module_name for target_key in qkv_modules_list):
                curr_params_list = qkv_params
            else:
                curr_params_list = proj_params

            if module.weight.requires_grad:
                curr_params_list.append(module.weight)
    for name, p in model.named_parameters():
        if (("norm" in name and proj_norms) or
            ("embed_tokens" in name and proj_embeds) or
            ("lm_head" in name and proj_logits and not lm_head_separate_group)):
            proj_params.append(p)

    if lm_head_separate_group:
        for name, p in model.named_parameters():
            if "lm_head" in name:
                lm_head_params.append(p)

    id_proj_params = [id(p) for p in proj_params] + [id(p) for p in lm_head_params] + [id(p) for p in qkv_params]
    regular_params = [p for p in model.parameters() if id(p) not in id_proj_params and p.requires_grad]
    
    param_groups = [{'params': regular_params, 'is_proj_params': False}, 
            {'params': proj_params, 'is_proj_params': True}]
    
    if lm_head_separate_group:
        param_groups.append({'params': lm_head_params, 'is_proj_params': proj_logits, 'is_lm_head': True})

    if qkv_separate_group:
        param_groups.append({'params': qkv_params, 'is_proj_params': True, 'is_qkv_params': True})

    return param_groups


class ProjOptimizer(Optimizer):

    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True, 
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

    ):
        assert isinstance(params, List) or proj_params is not None, "One should be either seperate proj params in the 'params' or explicitly pass them as 'proj_params'."
        assert 0.0 <= density <= 1.0
        assert proj_params_lr_scale >= 0
        assert inactive_lr_scale >= 0

        
        proj_params_args_dict = {'density': density, 'update_gap': update_gap, 'proj_params_lr_scale': proj_params_lr_scale, 'reset_statistics': reset_statistics, 
                                 'inactive_lr_scale': inactive_lr_scale, 'inactive_update_rule': inactive_update_rule, '_example_state_init': _example_state_init}
        if not isinstance(params, List):
            id_proj_params = [id(p) for p in proj_params]
            regular_params = [p for p in params if id(p) not in id_proj_params]
            return [{'params': regular_params, 'is_proj_params': False},
                    {'params': proj_params, 'is_proj_params': True}.update(proj_params_args_dict)]
        else:
            for group in params:
                assert isinstance(group, Dict)
                if group.get("is_proj_params", False):
                    for k, v in proj_params_args_dict.items():
                        group.setdefault(k, v)
                    group["step"] = 0
            return params


    def is_proj_group(self, group):
        return "is_proj_params" in group and group["is_proj_params"]


    @torch.no_grad()
    @abstractmethod
    def _update_states(self, group):
        pass


    @torch.no_grad()
    @abstractmethod
    def _proj_params_update(self, grad, state, group):
        pass


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
            if self.is_proj_group(group):
                if group["step"] % group["update_gap"] == 0:
                    self._update_states(group)
                group["step"] += 1
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ProjOptimizer does not support sparse gradients")
                
                state = self.state[p]
                
                if len(state) == 0:
                    self._init_state(example=p, state=state)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                if not self.is_proj_group(group):
                    update = self._compute_update(grad, state, **group)
                else:
                    update = self._proj_params_update(grad, state, group)

                p.add_(update)

        return loss
    

class GaloreOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

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
        )
        for group in params:
            if self.is_proj_group(group):
                group["proj_side"] = proj_side
                group["proj_type"] = proj_type
        return params
    

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            state = self.state[p]
            grad = p.grad
            if "projector" not in state:
                state["projector"] = GaLoreProjector(group["density"], grad_shape=grad.shape, proj_side=group["proj_side"], proj_type=group["proj_type"])
            state["projector"].update_proj(grad)
            if "step" not in state or group["reset_statistics"]:
                # reset
                if len(state) == 1 or group["_example_state_init"]:
                    self._init_state(example=state["projector"].project_down(grad), state=state)
                else:
                    self._init_state(state=state)

    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]
        update = self._compute_update(grad_down, state, **{**group, 'lr': active_lr})
        update = state["projector"].project_up(update)
        if group["inactive_update_rule"] == "no":
            return update
        inactive_grad = grad - state["projector"].project_up(grad_down)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            update.add_(-inactive_lr * inactive_grad)
        elif group["inactive_update_rule"] == "sign_sgd":
            update.add_(-inactive_lr * inactive_grad.sign())
        return update
    

class CoordOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,
        
        # coord specific
        coord_choice='columns'
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
        )
        for group in params:
            if self.is_proj_group(group):
                group["coord_choice"] = coord_choice
        return params
    

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            state = self.state[p]
            grad = p.grad
            if "projector" not in state:
                state["projector"] = CoordinateProjector(group["density"], grad_shape=grad.shape, coord_choice=group["coord_choice"])
            state["projector"].update_proj(grad)
            
            if "step" not in state or group["reset_statistics"]:
                # reset
                if len(state) == 1 or group["_example_state_init"]:
                    self._init_state(example=state["projector"].project_down(grad), state=state)
                else:
                    self._init_state(state=state)

    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]
        update = self._compute_update(grad_down, state, **{**group, 'lr': active_lr})
        update = state["projector"].project_up(update)
        if group["inactive_update_rule"] == "no":
            return update
        inactive_grad = grad - state["projector"].project_up(grad_down)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            update.add_(-inactive_lr * inactive_grad)
        elif group["inactive_update_rule"] == "sign_sgd":
            update.add_(-inactive_lr * inactive_grad.sign())
        return update
    

class BlockOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # block specific
        block_order='random',
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
        )
        # self.inactive_update_rule = inactive_update_rule
        for group in params:
            if self.is_proj_group(group):
                group["block_order"] = block_order
                group["num_active_blocks"] = round(len(group["params"]) * group["density"])
                assert not (group["block_order"] == "mirror" and group["num_active_blocks"] % 2), f"num tensors: {len(group['params'])}, num_active_blocks: {group['num_active_blocks']}"
        return params
    
    @torch.no_grad()
    def _activate_param(self, p):
        # p.requires_grad_(True)
        state = self.state[p]
        self._init_state(example=p, state=state)
        state["active"] = True

    @torch.no_grad()
    def _deactivate_param(self, p):
        state = self.state[p]
        state.clear()
        # state["step"] = 0
        state["active"] = False
        # if self.inactive_update_rule == "no":
        #     p.requires_grad_(False)

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            # if p.grad is None:
            #     continue
            # reset
            self._deactivate_param(p)
        if group["block_order"] == "random":
            current_blocks = torch.randperm(len(group["params"]))[:group["num_active_blocks"]]
            for idx in current_blocks:
                self._activate_param(group["params"][idx])
        elif group["block_order"] == "ascending":
            if "next_block_start" not in group:
                group["next_block_start"] = 0
            for idx in range(group["next_block_start"], group["next_block_start"] + group["num_active_blocks"]):
                self._activate_param(group["params"][idx % len(group["params"])])
            group["next_block_start"] += group["num_active_blocks"]
            if group["next_block_start"] >= len(group["params"]):
                group["next_block_start"] -= len(group["params"])
        elif group["block_order"] == "descending":
            if "next_block_start" not in group:
                group["next_block_start"] = len(group["params"]) - 1
            for idx in range(group["next_block_start"], group["next_block_start"] - group["num_active_blocks"], -1):
                self._activate_param(group["params"][idx % len(group["params"])])
            group["next_block_start"] -= group["num_active_blocks"]
            if group["next_block_start"] < 0:
                group["next_block_start"] += len(group["params"])
        elif group["block_order"] == "mirror":
            if "next_block_start" not in group:
                group["next_block_start"] = 0
            for idx in range(group["next_block_start"], group["next_block_start"] + group["num_active_blocks"] // 2):
                self._activate_param(group["params"][idx % (len(group["params"]) // 2)])
                self._activate_param(group["params"][len(group["params"]) - 1 - idx % (len(group["params"]) // 2)])
            group["next_block_start"] += group["num_active_blocks"] // 2
            if group["next_block_start"] >= len(group["params"]) // 2:
                group["next_block_start"] -= len(group["params"]) // 2
    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        if state["active"]:
            active_lr = group["lr"] * group["proj_params_lr_scale"]
            return self._compute_update(grad, state, **{**group, 'lr': active_lr})
        elif group["inactive_update_rule"] == "no":
            return torch.zeros_like(grad)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            return -inactive_lr * grad
        elif group["inactive_update_rule"] == "sign_sgd":
            return -inactive_lr * grad.sign()