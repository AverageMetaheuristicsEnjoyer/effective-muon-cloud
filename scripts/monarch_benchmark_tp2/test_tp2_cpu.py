"""CPU/Gloo correctness test for the TP=2 layout and Megatron Muon modes.

Run with:
  torchrun --standalone --nproc_per_node=2 -m scripts.monarch_benchmark_tp2.test_tp2_cpu
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .benchmark_tp2 import TP2MonarchLinear, all_to_all, attention_surrogate


def assert_close(actual: torch.Tensor, expected: torch.Tensor, description: str) -> None:
    error = (actual - expected).abs().max()
    if float(error.detach()) > 1e-5:
        raise AssertionError(f"{description}: maximum error is {float(error.detach())}")


def gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    pieces = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, tensor)
    return torch.cat(pieces, dim=-1)


def butterfly_reference_autograd(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The repository butterfly layout expressed with ordinary PyTorch ops."""
    batch, _ = x.shape
    blocks, _, p = left.shape
    _, s, r = right.shape
    first = torch.bmm(x.reshape(batch, blocks, p).transpose(0, 1), left.transpose(-1, -2))
    middle = first.transpose(0, 1).reshape(batch, r, blocks).transpose(-1, -2).contiguous()
    second = torch.bmm(middle.transpose(0, 1), right.transpose(-1, -2))
    return second.permute(1, 2, 0).reshape(batch, s * blocks)


def test_all_to_all_backward() -> None:
    """The autograd-aware all-to-all's reverse communication is its adjoint."""
    values = nn.Parameter(torch.zeros(2, 1))
    coefficients = torch.tensor([[10.0 * dist.get_rank()], [10.0 * dist.get_rank() + 1.0]])
    (all_to_all(values) * coefficients).sum().backward()
    expected = torch.tensor([[float(dist.get_rank())], [10.0 + float(dist.get_rank())]])
    assert_close(values.grad, expected, "All-to-all backward")


def test_monarch_permutation_backward() -> None:
    """Check the two layout transforms without either matrix multiply."""
    torch.manual_seed(200 + dist.get_rank())
    local = nn.Parameter(torch.randn(2, 4))
    packed = local.reshape(2, -1, 2).permute(2, 0, 1).contiguous()
    mixed = all_to_all(packed).permute(1, 0, 2).reshape_as(local)
    packed = mixed.reshape(2, 2, -1).permute(1, 0, 2).contiguous()
    output = all_to_all(packed).permute(1, 2, 0).reshape_as(local)
    full = gather_last_dim(local).detach().requires_grad_(True)
    middle = full.reshape(2, 4, 2).transpose(-1, -2)
    expected = middle.permute(0, 2, 1).reshape_as(full)
    assert_close(gather_last_dim(output), expected, "Monarch permutation forward")
    output.square().mean().backward()
    expected.chunk(2, dim=-1)[dist.get_rank()].square().mean().backward()
    dist.all_reduce(full.grad)
    assert_close(local.grad, full.grad.chunk(2, dim=-1)[dist.get_rank()], "Monarch permutation backward")


def test_monarch_layout() -> None:
    """Both all-to-alls must reproduce the repository's exact N=2 operator."""
    from models.monarch.monarch_linear import blockdiag_butterfly_multiply

    torch.manual_seed(100 + dist.get_rank())
    linear = TP2MonarchLinear(8, 24)
    torch.manual_seed(7)
    full_input = torch.randn(3, 8)
    local_input = full_input.chunk(dist.get_world_size(), dim=-1)[dist.get_rank()].contiguous()
    local_output = linear(local_input)
    actual = gather_last_dim(local_output)
    left = [torch.empty_like(linear.left) for _ in range(dist.get_world_size())]
    right = [torch.empty_like(linear.right) for _ in range(dist.get_world_size())]
    dist.all_gather(left, linear.left)
    dist.all_gather(right, linear.right)
    left_reference = torch.cat(left).detach().requires_grad_(True)
    right_reference = torch.cat(right).detach().requires_grad_(True)
    expected = blockdiag_butterfly_multiply(full_input, left_reference, right_reference)
    reference_autograd = butterfly_reference_autograd(full_input, left_reference, right_reference)
    assert_close(actual, expected, "Monarch forward layout")
    assert_close(reference_autograd, expected, "Reference butterfly forward")
    local_loss = local_output.square().mean()
    reference_local = reference_autograd.chunk(dist.get_world_size(), dim=-1)[dist.get_rank()]
    reference_loss = reference_local.square().mean()
    local_loss.backward()
    reference_loss.backward()
    # Each rank supplies one canonical output shard. Sum the two independent
    # reference losses to match all-to-all backward's cross-rank gradient flow.
    dist.all_reduce(left_reference.grad)
    dist.all_reduce(right_reference.grad)
    assert_close(
        linear.left.grad,
        left_reference.grad[dist.get_rank() : dist.get_rank() + 1],
        "Monarch left-factor gradient",
    )
    assert_close(
        linear.right.grad,
        right_reference.grad[dist.get_rank() : dist.get_rank() + 1],
        "Monarch right-factor gradient",
    )
    for parameter in linear.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError("Monarch backward did not produce finite gradients")


def test_megatron_tp_layers_and_muon_modes() -> None:
    """Exercise actual MCore TP layers and all three NVIDIA Muon modes on Gloo."""
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.optimizer.muon import TensorParallelMuon
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    config = ModelParallelConfig(
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )
    init = lambda weight: nn.init.uniform_(weight, -0.1, 0.1)
    common = {"config": config, "init_method": init, "bias": False, "tp_group": dist.group.WORLD}
    column = ColumnParallelLinear(8, 24, gather_output=False, **common)
    row = RowParallelLinear(8, 8, input_is_parallel=True, skip_bias_add=False, **common)
    torch.manual_seed(11)
    x = torch.randn(2, 3, 8)
    qkv, _ = column(x)
    output, _ = row(attention_surrogate(qkv))
    output.square().mean().backward()

    saved_weight = column.weight.detach().clone()
    saved_grad = column.weight.grad.detach().clone()
    for mode in ("duplicated", "distributed", "blockwise"):
        column.weight.data.copy_(saved_weight)
        column.weight.grad = saved_grad.clone()
        optimizer = TensorParallelMuon(
            [column.weight],
            lr=1e-4,
            momentum_beta=0.95,
            pg_collection=ProcessGroupCollection(tp=dist.group.WORLD),
            mode=mode,
            num_ns_steps=5,
        )
        optimizer.step()
        if not torch.isfinite(column.weight).all():
            raise AssertionError(f"Megatron TensorParallelMuon {mode} produced non-finite weights")


def main() -> None:
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise RuntimeError("this test requires exactly two ranks")
    try:
        test_all_to_all_backward()
        test_monarch_permutation_backward()
        test_monarch_layout()
        test_megatron_tp_layers_and_muon_modes()
        dist.barrier()
        if dist.get_rank() == 0:
            print("TP2_CPU_CORRECTNESS_OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
