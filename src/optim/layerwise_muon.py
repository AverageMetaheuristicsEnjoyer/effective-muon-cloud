import math
from collections import defaultdict

import torch

from third_party.lite import muonlite


class GroupedDenseMuon(muonlite.MuonLite):
    def __init__(self, *args, batch_size=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []
        for group in self.param_groups:
            shapes = defaultdict(list)
            for p in group["params"]:
                if self.state[p]["use_muon"] == 2:
                    self.state[p]["use_muon"] = 4
                    shapes[tuple(p.shape)].append(p)
            for parameters in shapes.values():
                for start in range(0, len(parameters), batch_size):
                    self.batches.append((group, parameters[start:start + batch_size]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group, parameters in self.batches:
            theta = group["muon_theta"]
            gradients = torch.stack([p.grad for p in parameters])
            buffers = []
            for p in parameters:
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                    state["step"] = 0
                buffers.append(state["momentum"])
                state["step"] += 1
            momentum = torch.stack(buffers) * theta + gradients * (1.0 - theta)
            torch._foreach_copy_(buffers, list(momentum.unbind()))
            direction = momentum + gradients * ((1.0 - theta) / theta)
            updates = muonlite.zeropower_via_newtonschulz5(direction, group["ns_steps"])
            torch._foreach_mul_(parameters, 1.0 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(parameters, list(updates.unbind()),
                                alpha=-0.2 * group["lr"] * math.sqrt(max(parameters[0].shape)))
        return loss


def make_dense_muon(model, implementation="reference", batch_size=4):
    muon, adamw = [], []
    for name, p in model.named_parameters():
        target = muon if p.ndim == 2 and not any(k in name for k in ("wte", "wpe", "lm_head", "embed", "core_logits")) else adamw
        target.append((name, p))
    options = dict(muon_params=muon, adamw_params=adamw, lr=1e-3, weight_decay=0.1,
                   ns_steps=6, muon_theta=0.95, adamw_betas=(0.9, 0.99), adamw_eps=1e-8,
                   total_steps=39250, warmup_steps=2000, beta1=0.0, beta2=0.0,
                   chi=1.0, chi_adamw=1.0, subspace_ratio=0.0)
    if implementation == "grouped":
        return GroupedDenseMuon(**options, batch_size=batch_size)
    return muonlite.MuonLite(**options)
