from __future__ import annotations

import json
import os
import subprocess

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models.monarch import MonarchLinear, patch_monarch_linear
from models.monarch.monarch_linear import blockdiag_butterfly_multiply
from models.monarch.monarch_ops import butterfly_blk
from models.monarch.monarch_muon import MonarchMuonOptimizer, _newton_schulz_batched


def gpu_uuid(index: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    )
    return output.splitlines()[index].strip()


def check_fast_riffle(device: torch.device) -> dict[int, float]:
    patch_monarch_linear(blocked=True, fast_riffle=True)
    errors = {}
    for blocks in (2, 4):
        torch.manual_seed(100 + blocks)
        x = torch.randn(32, blocks * 64, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(blocks, 64, 64, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(blocks, 64, 64, device=device, dtype=torch.bfloat16)
        grad = torch.randn_like(x)

        ref_tensors = [tensor.detach().clone().requires_grad_() for tensor in (x, w1, w2)]
        fast_tensors = [tensor.detach().clone().requires_grad_() for tensor in (x, w1, w2)]
        reference = blockdiag_butterfly_multiply(*ref_tensors)
        fast = butterfly_blk(*fast_tensors)
        reference.backward(grad)
        fast.backward(grad)

        error = max(
            [(reference - fast).abs().max().item()]
            + [
                (left.grad - right.grad).abs().max().item()
                for left, right in zip(ref_tensors, fast_tensors)
            ]
        )
        if error != 0.0:
            raise RuntimeError(f"N={blocks} fast riffle mismatch: max_abs_error={error}")
        errors[blocks] = error
    return errors


def check_muon_semantics(device: torch.device) -> float:
    torch.manual_seed(314)
    parameter = torch.nn.Parameter(torch.randn(2, 8, 12, device=device))
    gradient = torch.randn_like(parameter)
    initial = parameter.detach().clone()
    lr, momentum, weight_decay = 1e-2, 0.9, 0.1
    expected_update = _newton_schulz_batched(gradient * (1.0 + momentum))
    expected = initial * (1.0 - lr * weight_decay)
    expected.add_(expected_update, alpha=-lr * (8 / 12) ** 0.5)

    parameter.grad = gradient
    optimizer = MonarchMuonOptimizer(
        [parameter],
        [],
        lr=lr,
        momentum=momentum,
        nesterov=True,
        muon_weight_decay=weight_decay,
        ns_dtype=torch.float32,
    )
    optimizer.step()
    error = (parameter - expected).abs().max().item()
    if error > 1e-6:
        raise RuntimeError(f"Muon Nesterov/weight-decay mismatch: max_abs_error={error}")
    return error


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= rank < world_size or not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(f"invalid rank mapping rank={rank} local_rank={local_rank} world={world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "gpu_uuid": gpu_uuid(local_rank),
    }
    identities = [None] * world_size
    dist.all_gather_object(identities, identity)
    if rank == 0:
        if {item["rank"] for item in identities} != set(range(world_size)):
            raise RuntimeError(f"duplicate or missing ranks: {identities}")
        for field in ("local_rank", "pid", "gpu_uuid"):
            if len({item[field] for item in identities}) != world_size:
                raise RuntimeError(f"duplicate {field}: {identities}")

    errors = check_fast_riffle(device)
    muon_error = check_muon_semantics(device)

    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        MonarchLinear(128, 128, bias=False, nblocks=2, device=device, dtype=torch.bfloat16),
        torch.nn.GELU(),
        MonarchLinear(128, 128, bias=False, nblocks=4, device=device, dtype=torch.bfloat16),
    )
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(2000 + rank)
    inputs = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    loss = model(inputs).float().square().mean()
    loss.backward()
    optimizer.step()

    for parameter in model.parameters():
        copies = [torch.empty_like(parameter) for _ in range(world_size)]
        dist.all_gather(copies, parameter)
        if any(not torch.equal(copies[0], other) for other in copies[1:]):
            raise RuntimeError("DDP parameters diverged after one optimizer step")

    dist.barrier()
    if rank == 0:
        print(
            "MONARCH_CLOUD_SMOKE="
            + json.dumps(
                {
                    "world_size": world_size,
                    "processes": identities,
                    "fast_riffle_max_abs_error": errors,
                    "muon_semantics_max_abs_error": muon_error,
                    "ddp_parameters_synchronized": True,
                    "nested_torchrun": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
