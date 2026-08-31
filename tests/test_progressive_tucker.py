from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


WORKSPACE = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = Path(__file__).resolve().parents[1] / "src"
FULL_SRC = WORKSPACE / "Tucker experiment" / "src"
for path in (str(FULL_SRC), str(REPOSITORY_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.tucker_linear import TuckerLinear
from optim.progressive_tucker import (
    ProgressiveTuckerController,
    expand_tucker_model_to_plan_,
    restore_progressive_tucker_shapes_,
    shrink_tucker_model_to_plan_,
)
from optim.tensorion import TensorionOptimizer


class TinyTuckerModel(nn.Module):
    def __init__(self, ranks=(2, 2, 2, 2)):
        super().__init__()
        self.layer = TuckerLinear(
            16,
            16,
            rank=ranks,
            bias=False,
            equal_params=False,
            forward_mode="materialize",
            dtype=torch.float64,
        )


class ProgressiveTuckerCudaRegression(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for TF32")
    def test_function_check_disables_tf32_and_restores_backend_state(self):
        torch.manual_seed(20260828)
        model = nn.Module().cuda()
        model.layer = TuckerLinear(
            1024,
            2816,
            rank=(29, 30, 25, 36),
            bias=False,
            equal_params=False,
            forward_mode="materialize",
            device="cuda",
            dtype=torch.float32,
        )
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            metrics = expand_tucker_model_to_plan_(
                model,
                None,
                {"layer": (30, 31, 31, 44)},
                seed=1_001_704,
                verify_function=True,
                verify_rtol=5e-5,
            )
            self.assertLessEqual(metrics["max_relative_function_error"], 5e-5)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


def _populate_sgd_momentum(model, optimizer):
    inputs = torch.randn(5, 16, dtype=torch.float64)
    model.layer(inputs).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_rank_expansion_preserves_function_and_optimizer_state():
    torch.manual_seed(7)
    model = TinyTuckerModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    _populate_sgd_momentum(model, optimizer)
    # Production runs retract every optimizer step; rank growth relies on that
    # Stiefel invariant when it samples an orthogonal complement.
    model.layer.retract_()

    parameters_before = {
        name: parameter for name, parameter in model.named_parameters()
    }
    weight_before = model.layer.materialize_weight(dtype=torch.float64)
    old_core_momentum = optimizer.state[model.layer.core_matrix][
        "momentum_buffer"
    ].reshape(2, 2, 2, 2).clone()

    metrics = expand_tucker_model_to_plan_(
        model,
        optimizer,
        {"layer": (4, 4, 4, 4)},
        seed=11,
        verify_function=True,
        verify_rtol=1e-6,
    )

    assert metrics["expanded_modules"] == 1
    assert model.layer.ranks == (4, 4, 4, 4)
    assert torch.allclose(
        weight_before,
        model.layer.materialize_weight(dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )
    for name, parameter in model.named_parameters():
        assert parameter is not parameters_before[name]
        assert any(
            parameter is grouped
            for group in optimizer.param_groups
            for grouped in group["params"]
        )

    new_core_momentum = optimizer.state[model.layer.core_matrix][
        "momentum_buffer"
    ].reshape(4, 4, 4, 4)
    assert torch.equal(
        new_core_momentum[:2, :2, :2, :2],
        old_core_momentum,
    )
    outside = new_core_momentum.clone()
    outside[:2, :2, :2, :2] = 0
    assert torch.count_nonzero(outside) == 0

    for factor in (model.layer.U1, model.layer.U2, model.layer.U3, model.layer.U4):
        identity = torch.eye(4, dtype=factor.dtype)
        assert torch.allclose(factor.mT @ factor, identity, atol=1e-6, rtol=1e-6)


def test_progressive_checkpoint_restores_shapes_before_state_load():
    torch.manual_seed(13)
    model = TinyTuckerModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    current = sum(parameter.numel() for parameter in model.parameters())
    full_plan = {"layer": (4, 4, 4, 4)}
    full = current - model.layer.tucker_parameter_count + (4 * 4 * 4 * 4 + 64)
    controller = ProgressiveTuckerController(
        model,
        optimizer,
        [f"0:{current}", f"1:{full}"],
        warmup_steps=3,
        seed=17,
        verify_rtol=1e-6,
    )
    controller.maybe_grow(1)
    _populate_sgd_momentum(model, optimizer)
    saved_model = model.state_dict()
    saved_optimizer = optimizer.state_dict()
    saved_progressive = dict(model._progressive_tucker_state)

    restored = TinyTuckerModel()
    restored_optimizer = torch.optim.SGD(
        restored.parameters(), lr=1e-2, momentum=0.9
    )
    restore_progressive_tucker_shapes_(
        restored,
        restored_optimizer,
        saved_progressive,
    )
    restored.load_state_dict(saved_model)
    restored_optimizer.load_state_dict(saved_optimizer)

    assert restored.layer.ranks == full_plan["layer"]
    assert torch.equal(
        restored.layer.materialize_weight(dtype=torch.float64),
        model.layer.materialize_weight(dtype=torch.float64),
    )
    assert restored_optimizer.state[restored.layer.core_matrix][
        "momentum_buffer"
    ].shape == restored.layer.core_matrix.shape


def test_rank_shrink_projects_weight_and_optimizer_state():
    torch.manual_seed(17)
    model = TinyTuckerModel(ranks=(4, 4, 4, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    _populate_sgd_momentum(model, optimizer)
    model.layer.retract_()

    metrics = shrink_tucker_model_to_plan_(
        model,
        optimizer,
        {"layer": (2, 2, 2, 2)},
    )

    assert metrics["shrunk_modules"] == 1
    assert metrics["direction"] == "shrink"
    assert model.layer.ranks == (2, 2, 2, 2)
    assert torch.isfinite(torch.tensor(metrics["max_relative_function_error"]))
    for factor in (model.layer.U1, model.layer.U2, model.layer.U3, model.layer.U4):
        assert optimizer.state[factor]["momentum_buffer"].shape == factor.shape
        assert torch.allclose(
            factor.mT @ factor,
            torch.eye(2, dtype=factor.dtype),
            atol=1e-6,
            rtol=1e-6,
        )
    assert optimizer.state[model.layer.core_matrix][
        "momentum_buffer"
    ].shape == model.layer.core_matrix.shape


def test_reverse_progressive_controller_uses_requested_final_ranks():
    torch.manual_seed(23)
    model = TinyTuckerModel(ranks=(4, 4, 4, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    full = sum(parameter.numel() for parameter in model.parameters())
    low = full - model.layer.tucker_parameter_count + (
        2 * 2 * 2 * 2 + sum(mode * 2 for mode in model.layer.modes)
    )
    controller = ProgressiveTuckerController(
        model,
        optimizer,
        [f"0:{full}", f"1:{low}"],
        final_ranks="2,2,2,2",
        warmup_steps=0,
    )

    transition = controller.maybe_resize(1)

    assert transition is not None
    assert transition["direction"] == "shrink"
    assert transition["actual_parameters"] == low
    assert model.layer.ranks == (2, 2, 2, 2)

    _populate_sgd_momentum(model, optimizer)
    saved_model = model.state_dict()
    saved_optimizer = optimizer.state_dict()
    saved_progressive = dict(model._progressive_tucker_state)
    restored = TinyTuckerModel(ranks=(4, 4, 4, 4))
    restored_optimizer = torch.optim.SGD(
        restored.parameters(), lr=1e-2, momentum=0.9
    )
    restore_progressive_tucker_shapes_(
        restored,
        restored_optimizer,
        saved_progressive,
    )
    restored.load_state_dict(saved_model)
    restored_optimizer.load_state_dict(saved_optimizer)
    assert restored.layer.ranks == (2, 2, 2, 2)
    assert restored_optimizer.state[restored.layer.core_matrix][
        "momentum_buffer"
    ].shape == restored.layer.core_matrix.shape


def test_tensorion_can_step_and_retract_after_growth():
    torch.manual_seed(19)
    model = TinyTuckerModel()
    layer = model.layer
    factors = (layer.U1, layer.U2, layer.U3, layer.U4)
    optimizer = TensorionOptimizer(
        tensorion_params=[("layer.core_matrix", layer.core_matrix, (2, 2, 2, 2))],
        adamw_param_groups=[],
        riemannian_muon_params=[
            (f"layer.U{index}", factor)
            for index, factor in enumerate(factors, start=1)
        ],
        tucker_module_specs=[("layer", layer.core_matrix, factors)],
        tucker_lr_scaling_mode="first_order_calibrated",
        tucker_lr_scaling_post_ns_project=False,
        lr=1e-3,
        momentum=0.9,
        adjust_lr=True,
        ns_steps=2,
    )

    inputs = torch.randn(4, 16, dtype=torch.float64)
    layer(inputs).square().mean().backward()
    optimizer.step()
    layer.retract_with_optimizer_state_(optimizer)
    optimizer.zero_grad(set_to_none=True)

    expand_tucker_model_to_plan_(
        model,
        optimizer,
        {"layer": (4, 4, 4, 4)},
        seed=23,
        verify_rtol=1e-6,
    )
    assert optimizer._plans[layer.core_matrix].tensor_shape == (4, 4, 4, 4)
    factors = (layer.U1, layer.U2, layer.U3, layer.U4)
    for factor in factors:
        assert optimizer.state[factor]["momentum_buffer"].shape == factor.shape

    layer(inputs).square().mean().backward()
    optimizer.step()
    layer.retract_with_optimizer_state_(optimizer)
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(layer.core_matrix).all()
