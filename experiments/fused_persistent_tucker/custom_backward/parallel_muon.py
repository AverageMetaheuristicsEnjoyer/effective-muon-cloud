"""Stream-parallel grouped Muon for Tucker cores and mode factors.

All mathematical updates are identical to vanilla MuonLite.  The only change
is scheduling: equal-shaped factors are processed as batches, independent
Tucker cores are distributed between worker CUDA streams, and the original
AdamW path runs concurrently on the caller's stream (including the Dense
lm_head).  A production A100 sweep selected two worker streams and individual
core updates; batching the comparatively large cores was slower.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from third_party.lite.muonlite import MuonLite, zeropower_via_newtonschulz5


class ParallelGroupedMuonLite(MuonLite):
    """Batch and overlap independent vanilla-Muon matrix updates."""

    def __init__(
        self,
        *args,
        core_microbatch: int = 1,
        factor_microbatch: int = 0,
        parallel_streams: int = 2,
        **kwargs,
    ):
        if core_microbatch <= 0:
            raise ValueError("core_microbatch must be positive")
        if factor_microbatch < 0:
            raise ValueError("factor_microbatch must be non-negative")
        if parallel_streams <= 0:
            raise ValueError("parallel_streams must be positive")
        super().__init__(*args, **kwargs)
        self.core_microbatch = int(core_microbatch)
        self.factor_microbatch = int(factor_microbatch)
        self.parallel_streams = int(parallel_streams)
        self._streams_by_device = {}

        grouped = defaultdict(list)
        for group_index, group in enumerate(self.param_groups):
            for parameter in group["params"]:
                state = self.state[parameter]
                if state.get("use_muon") != 2:
                    continue
                name = state.get("name", "")
                if ".core_matrix" in name:
                    kind = "core"
                elif any(marker in name for marker in (".U1", ".U2", ".U3", ".U4")):
                    kind = "factor"
                else:
                    continue
                state["use_muon"] = 4
                grouped[
                    (group_index, kind, tuple(parameter.shape), parameter.device)
                ].append(parameter)

        self._parallel_groups = []
        for (group_index, kind, shape, _device), parameters in grouped.items():
            chunk_size = (
                self.core_microbatch
                if kind == "core"
                else self.factor_microbatch or len(parameters)
            )
            chunks = [
                parameters[start : start + chunk_size]
                for start in range(0, len(parameters), chunk_size)
            ]
            self._parallel_groups.append(
                {
                    "group_index": group_index,
                    "kind": kind,
                    "shape": shape,
                    "chunks": chunks,
                }
            )

    def _mark_parallel_parameters(self):
        for group in self._parallel_groups:
            for chunk in group["chunks"]:
                for parameter in chunk:
                    self.state[parameter]["use_muon"] = 4

    def state_dict(self):
        """Write standard-Muon tags so checkpoints remain backward compatible."""
        result = super().state_dict()
        # Optimizer.state_dict() does not deep-copy tensor values.  Copy only
        # the tiny per-parameter dictionaries before changing the routing tag;
        # copying momentum tensors here would temporarily duplicate >1 GiB.
        result["state"] = {
            key: dict(parameter_state)
            for key, parameter_state in result["state"].items()
        }
        for parameter_state in result["state"].values():
            if parameter_state.get("use_muon") == 4:
                parameter_state["use_muon"] = 2
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        # Vanilla checkpoints contain use_muon=2.  Restore the transient route
        # or the parent step would update the same parameter a second time.
        self._mark_parallel_parameters()
        return result

    @property
    def grouped_core_count(self) -> int:
        return sum(
            len(chunk)
            for group in self._parallel_groups
            if group["kind"] == "core"
            for chunk in group["chunks"]
        )

    @property
    def grouped_factor_count(self) -> int:
        return sum(
            len(chunk)
            for group in self._parallel_groups
            if group["kind"] == "factor"
            for chunk in group["chunks"]
        )

    def _streams(self, device):
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        if index not in self._streams_by_device:
            self._streams_by_device[index] = [
                torch.cuda.Stream(device=index) for _ in range(self.parallel_streams)
            ]
        return self._streams_by_device[index]

    def _update_batch(self, parameters, *, lr, weight_decay, momentum_decay, ns_steps):
        momentums = []
        gradients = []
        active_parameters = []
        for parameter in parameters:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(parameter.grad)
            active_parameters.append(parameter)
            momentums.append(state["momentum"])
            gradients.append(parameter.grad)
        if not active_parameters:
            return

        torch._foreach_mul_(momentums, momentum_decay)
        torch._foreach_add_(momentums, gradients, alpha=1.0 - momentum_decay)
        nesterov = torch._foreach_add(
            momentums,
            gradients,
            alpha=(1.0 - momentum_decay) / momentum_decay,
        )
        batched = torch.stack(nesterov)
        updates = zeropower_via_newtonschulz5(batched, ns_steps).unbind(0)
        torch._foreach_mul_(active_parameters, 1.0 - lr * weight_decay)
        torch._foreach_add_(
            active_parameters,
            updates,
            alpha=-0.2 * lr * math.sqrt(max(active_parameters[0].shape)),
        )
        for parameter in active_parameters:
            self.state[parameter]["step"] += 1

    @torch.no_grad()
    def step(self, closure=None):
        closure_loss = None
        if closure is not None:
            with torch.enable_grad():
                closure_loss = closure()

        used_streams = []
        core_stream_cursor = 0
        for scheduled_group in self._parallel_groups:
            optimizer_group = self.param_groups[scheduled_group["group_index"]]
            chunks = scheduled_group["chunks"]
            if not chunks:
                continue
            device = chunks[0][0].device
            streams = self._streams(device)
            caller_stream = torch.cuda.current_stream(device)
            for stream in streams:
                stream.wait_stream(caller_stream)

            if scheduled_group["kind"] == "factor":
                stream_index = min(len(streams) - 1, 3)
            else:
                stream_index = core_stream_cursor % min(len(streams), 3)
                core_stream_cursor += 1
            stream = streams[stream_index]
            used_streams.append((caller_stream, stream))
            for parameters in chunks:
                with torch.cuda.stream(stream):
                    self._update_batch(
                        parameters,
                        lr=optimizer_group["lr"],
                        weight_decay=optimizer_group["weight_decay"],
                        momentum_decay=optimizer_group["muon_theta"],
                        ns_steps=optimizer_group["ns_steps"],
                    )

        # Muon work is queued on worker streams.  The parent now processes only
        # untouched fallback matrices and AdamW parameters on the caller stream.
        super().step(None)
        for caller_stream, worker_stream in used_streams:
            caller_stream.wait_stream(worker_stream)
        return closure_loss
