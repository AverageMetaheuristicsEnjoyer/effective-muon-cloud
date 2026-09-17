import math
from collections import defaultdict

import torch

from models.layerwise_tucker import AttentionBank, LayerwiseTuckerMLP
from models.tucker_linear import _mode_product, qr_retract_with_transport
from optim.tensorion import TensorionOptimizer, fold_tensor, unfold_tensor


def layerwise_tucker_specs(model):
    specs = []
    for name, module in model.named_modules():
        if isinstance(module, AttentionBank):
            # The head axis of the core has no factor and is never retracted.
            specs.append((name, module.core, (module.hidden, module.head, module.role)))
        elif isinstance(module, LayerwiseTuckerMLP):
            specs.append((name, module.core, (module.ff, module.model, module.role)))
            if module.variant == "A":
                specs.append((name + ".down", module.down_core, (module.down_out, module.down_ff)))
    return specs


@torch.no_grad()
def retract_layerwise_reference(specs, optimizer=None):
    for _, core, factors in specs:
        value = core
        core_momentum = None if optimizer is None else optimizer.state[core]["momentum_buffer"]
        for mode, factor in enumerate(factors):
            momentum = None if optimizer is None else optimizer.state[factor]["momentum_buffer"]
            q, r, dq, dr = qr_retract_with_transport(factor, momentum)
            previous = value
            value = _mode_product(r, previous, mode)
            if optimizer is not None:
                core_momentum = _mode_product(r, core_momentum, mode) + _mode_product(dr, previous, mode)
                momentum.copy_(dq)
            factor.copy_(q)
        core.copy_(value)
        if optimizer is not None:
            optimizer.state[core]["momentum_buffer"].copy_(core_momentum)


def batched_mode_product(matrix, tensor, mode):
    moved = tensor.movedim(mode + 1, 1)
    return (matrix @ moved.flatten(2)).reshape(moved.shape).movedim(1, mode + 1)


@torch.no_grad()
def retract_layerwise_grouped(specs, optimizer, batch_size=4):
    groups = defaultdict(list)
    for spec in specs:
        groups[(tuple(spec[1].shape), tuple(tuple(p.shape) for p in spec[2]))].append(spec)
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            cores = [s[1] for s in chunk]
            value = torch.stack(cores)
            core_buffers = [optimizer.state[p]["momentum_buffer"] for p in cores]
            direction = torch.stack(core_buffers)
            for mode in range(len(chunk[0][2])):
                factors = [s[2][mode] for s in chunk]
                buffers = [optimizer.state[p]["momentum_buffer"] for p in factors]
                p, momentum = torch.stack(factors), torch.stack(buffers)
                q, r = torch.linalg.qr(p, mode="reduced")
                signs = torch.sign(r.diagonal(dim1=-2, dim2=-1))
                signs = torch.where(signs == 0, 1, signs)
                q, r = q * signs.unsqueeze(-2), r * signs.unsqueeze(-1)
                cross = q.mT @ momentum
                quotient = torch.linalg.solve_triangular(r.mT, cross.mT, upper=False).mT
                lower = torch.tril(quotient, diagonal=-1)
                omega = lower - lower.mT
                normal = momentum - q @ cross
                dq = torch.linalg.solve_triangular(r.mT, normal.mT, upper=False).mT + q @ omega
                dr = torch.triu(cross - omega @ r)
                direction = batched_mode_product(r, direction, mode) + batched_mode_product(dr, value, mode)
                value = batched_mode_product(r, value, mode)
                torch._foreach_copy_(factors, list(q.unbind()))
                torch._foreach_copy_(buffers, list(dq.unbind()))
            torch._foreach_copy_(cores, list(value.unbind()))
            torch._foreach_copy_(core_buffers, list(direction.unbind()))


def batched_tensorion_ns(matrix, steps):
    work = matrix / (torch.linalg.vector_norm(matrix, dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = work @ work.mT
        work = 3.4445 * work - 4.7750 * (gram @ work) + 2.0315 * ((gram @ gram) @ work)
    return work


class GroupedRiemannianOptimizer(TensorionOptimizer):
    def __init__(self, *args, batch_size=4, compile_updates=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.ns = torch.compile(batched_tensorion_ns, dynamic=True, fullgraph=True) if compile_updates else batched_tensorion_ns
        self.batches = {}
        for group in self.param_groups:
            if group["update_type"] == "adamw":
                continue
            groups = defaultdict(list)
            for parameter in group["params"]:
                groups[tuple(parameter.shape)].append(parameter)
            self.batches[id(group)] = [
                parameters[start:start + batch_size]
                for parameters in groups.values()
                for start in range(0, len(parameters), batch_size)
            ]

    def _update_group(self, group, core):
        for parameters in self.batches[id(group)]:
            points = torch.stack(parameters)
            gradients = torch.stack([p.grad for p in parameters])
            if not core:
                cross = points.mT @ gradients
                gradients = gradients - points @ (0.5 * (cross + cross.mT))
            buffers = []
            for parameter in parameters:
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffers.append(state["momentum_buffer"])
            momentum = torch.stack(buffers).mul_(self.momentum)
            momentum.add_(gradients, alpha=1.0 if core else 1.0 - self.momentum)
            torch._foreach_copy_(buffers, list(momentum.unbind()))
            if not core:
                momentum = momentum * self.momentum + gradients * (1.0 - self.momentum)
            plan = (self._plans if core else self._muon_plans)[parameters[0]]
            matrices = torch.stack([unfold_tensor(m, plan) for m in momentum.unbind()])
            updates = [fold_tensor(u, plan) for u in self.ns(matrices, self.ns_steps).unbind()]
            torch._foreach_mul_(parameters, 1.0 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(parameters, updates, alpha=-group["lr"] * 0.2 * math.sqrt(max(plan.rows, plan.columns)))

    def _tensorion_step(self, group):
        self._update_group(group, core=True)

    def _muon_step(self, group, *, project_gradient=False):
        self._update_group(group, core=False)


def make_layerwise_riemannian(model, implementation="reference", batch_size=4):
    specs = layerwise_tucker_specs(model)
    names = {p: name for name, p in model.named_parameters()}
    cores = [(names[core], core, tuple(core.shape)) for _, core, _ in specs]
    factors = [(names[p], p) for _, _, fs in specs for p in fs]
    tensorized = {p for _, p, _ in cores} | {p for _, p in factors}
    adamw = [p for p in model.parameters() if p not in tensorized]
    # Match the last production launcher: heavy-ball cores, no coupled LR
    # calibration, no post-NS projection, vector transport after every step.
    options = dict(tensorion_params=cores, riemannian_muon_params=factors,
                   adamw_param_groups=[dict(params=[p for p in adamw if p.ndim > 1], weight_decay=0.1),
                                       dict(params=[p for p in adamw if p.ndim == 1], weight_decay=0.0)],
                   lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=False,
                   adjust_lr=True, ns_steps=6, adamw_betas=(0.9, 0.99), adamw_eps=1e-8)
    if implementation == "grouped":
        optimizer = GroupedRiemannianOptimizer(**options, batch_size=batch_size,
                                             compile_updates=next(model.parameters()).is_cuda)
    else:
        optimizer = TensorionOptimizer(**options)
    return optimizer, specs
