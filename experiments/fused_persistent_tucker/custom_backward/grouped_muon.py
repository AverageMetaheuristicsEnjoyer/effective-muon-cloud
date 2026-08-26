"""MuonLite variant batching Newton--Schulz for equal-shaped small factors.

The 84 Tucker modules retain the same large core matrices as the Dense model,
but add four small mode factors apiece.  Running Newton--Schulz independently
for those factors creates hundreds of tiny GEMM graphs.  This class changes
only their scheduling: equal-shaped factors are stacked, orthogonalized as a
batch, and updated with foreach kernels.  Large cores and AdamW parameters
(including the Dense lm_head) continue through the unmodified MuonLite path.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from third_party.lite.muonlite import MuonLite, zeropower_via_newtonschulz5


class GroupedSmallFactorMuonLite(MuonLite):
    """Batch exact vanilla-Muon updates for repeated small matrix shapes."""

    def __init__(self, *args, small_factor_max_numel: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        grouped = defaultdict(list)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if (
                    state.get("use_muon") == 2
                    and parameter.numel() <= small_factor_max_numel
                    and any(
                        marker in state.get("name", "")
                        for marker in (".U1", ".U2", ".U3", ".U4")
                    )
                ):
                    grouped[(id(group), tuple(parameter.shape))].append(parameter)

        self._small_factor_groups = []
        for (_, shape), parameters in grouped.items():
            if len(parameters) < 2:
                continue
            for parameter in parameters:
                self.state[parameter]["use_muon"] = 3
            self._small_factor_groups.append((shape, parameters))

    @property
    def grouped_small_factor_count(self) -> int:
        return sum(len(parameters) for _, parameters in self._small_factor_groups)

    @torch.no_grad()
    def step(self, closure=None):
        # The parent ignores use_muon=3, while preserving the exact production
        # path for large cores and all AdamW parameters.
        loss = super().step(closure)

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum_decay = group["muon_theta"]
            ns_steps = group["ns_steps"]

            group_parameter_ids = {id(parameter) for parameter in group["params"]}
            for shape, candidates in self._small_factor_groups:
                parameters = [
                    parameter
                    for parameter in candidates
                    if id(parameter) in group_parameter_ids and parameter.grad is not None
                ]
                if not parameters:
                    continue

                momentums = []
                gradients = []
                for parameter in parameters:
                    state = self.state[parameter]
                    if "step" not in state:
                        state["step"] = 0
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(parameter.grad)
                    momentums.append(state["momentum"])
                    gradients.append(parameter.grad)

                torch._foreach_mul_(momentums, momentum_decay)
                torch._foreach_add_(
                    momentums, gradients, alpha=1.0 - momentum_decay
                )
                nesterov = torch._foreach_add(
                    momentums,
                    gradients,
                    alpha=(1.0 - momentum_decay) / momentum_decay,
                )
                batched = torch.stack(nesterov)
                updates = zeropower_via_newtonschulz5(batched, ns_steps).unbind(0)

                torch._foreach_mul_(parameters, 1.0 - lr * weight_decay)
                torch._foreach_add_(
                    parameters,
                    updates,
                    alpha=-0.2 * lr * math.sqrt(max(shape)),
                )
                for parameter in parameters:
                    self.state[parameter]["step"] += 1

        return loss

