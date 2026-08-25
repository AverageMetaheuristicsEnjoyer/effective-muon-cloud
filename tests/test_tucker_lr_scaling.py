import copy
import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optim.tensorion import TensorionOptimizer
from optim.tucker_lr_scaling import (
    tucker_paper_mup_lr_multipliers,
    tucker_spectral_denominator,
    two_factor_spectron_denominator,
    warm_started_spectral_norm,
)


def _orthogonal(rows, columns, *, dtype=torch.float64):
    value, _ = torch.linalg.qr(
        torch.randn(rows, columns, dtype=dtype),
        mode="reduced",
    )
    return value


def _dense_tucker(core, factors):
    U1, U2, U3, U4 = factors
    return torch.kron(U3, U4) @ core @ torch.kron(U1, U2).mT


class TuckerLearningRateScalingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_two_factor_reduces_to_spectron_formula(self):
        sigma_a = 2.5
        sigma_b = 4.0
        actual = two_factor_spectron_denominator(sigma_a, sigma_b)
        expected = sigma_a + sigma_b + 1.0
        torch.testing.assert_close(actual, torch.tensor(expected))

    def test_paper_mup_tucker_multipliers_follow_local_input_widths(self):
        core = torch.zeros(35, 6)
        factors = (
            torch.zeros(4, 2),
            torch.zeros(6, 3),
            torch.zeros(8, 5),
            torch.zeros(9, 7),
        )
        actual = tucker_paper_mup_lr_multipliers(core, factors)
        expected = (
            24 / (5 * 6),
            24 / (5 * 4),
            24 / (5 * 6),
            24 / (5 * 5),
            24 / (5 * 7),
        )
        self.assertEqual(actual, expected)

    def test_paper_mup_applies_distinct_component_lrs(self):
        core = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
        factors = tuple(
            torch.nn.Parameter(torch.ones(rows, 1, dtype=torch.float64))
            for rows in (2, 3, 4, 5)
        )
        core.grad = torch.ones_like(core)
        for factor in factors:
            factor.grad = torch.ones_like(factor)
        optimizer = TensorionOptimizer(
            tensorion_params=[("core", core, (1, 1, 1, 1))],
            riemannian_muon_params=[
                (f"U{index}", factor) for index, factor in enumerate(factors, start=1)
            ],
            adamw_param_groups=[],
            tucker_module_specs=[("layer", core, factors)],
            tucker_lr_scaling_mode="paper_mup",
            tucker_lr_scaling_log_interval=1,
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )
        optimizer.step()
        metrics = optimizer.last_tucker_lr_scaling_metrics
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/paper_kappa_core"], 1.2)
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/paper_kappa_U1"], 0.6)
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/paper_kappa_U2"], 0.4)
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/paper_kappa_U3"], 1.2)
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/paper_kappa_U4"], 1.2)

    def test_three_factor_bound_controls_ugv_update(self):
        U = torch.randn(6, 3, dtype=torch.float64)
        G = torch.randn(3, 4, dtype=torch.float64)
        V = torch.randn(5, 4, dtype=torch.float64)
        directions = [torch.randn_like(U), torch.randn_like(G), torch.randn_like(V)]
        factors = [U, G, V]
        sigmas = [torch.linalg.matrix_norm(value, ord=2) for value in factors]
        rhos = [torch.linalg.matrix_norm(value, ord=2) for value in directions]
        denominator = tucker_spectral_denominator(
            sigmas,
            rhos,
            mode="spectron_bound",
        )
        eta = 0.05
        alpha = min(1.0, eta / float(denominator))
        before = U @ G @ V.mT
        after = (U - alpha * directions[0]) @ (G - alpha * directions[1]) @ (
            V - alpha * directions[2]
        ).mT
        actual_delta = torch.linalg.matrix_norm(after - before, ord=2)
        self.assertLessEqual(float(actual_delta), eta + 1e-10)

    def test_rectangular_stiefel_five_factor_strict_bound(self):
        core = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float64))
        factors = tuple(
            torch.nn.Parameter(value)
            for value in (
                _orthogonal(3, 2),
                _orthogonal(4, 2),
                _orthogonal(5, 2),
                _orthogonal(3, 2),
            )
        )
        before = _dense_tucker(core.detach().clone(), [x.detach().clone() for x in factors])
        core.grad = torch.randn_like(core)
        for factor in factors:
            factor.grad = torch.randn_like(factor)

        optimizer = TensorionOptimizer(
            tensorion_params=[("core", core, (2, 2, 2, 2))],
            riemannian_muon_params=[
                (f"U{index}", factor) for index, factor in enumerate(factors, start=1)
            ],
            adamw_param_groups=[],
            tucker_module_specs=[("layer", core, factors)],
            tucker_lr_scaling_mode="spectron_bound",
            tucker_lr_scaling_exact_svd_debug=True,
            tucker_lr_scaling_strict_bound_check=True,
            tucker_lr_scaling_log_interval=1,
            lr=0.05,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )
        optimizer.step()
        after = _dense_tucker(core, factors)
        actual_delta = torch.linalg.matrix_norm(after - before, ord=2)
        self.assertLessEqual(float(actual_delta.detach()), 0.05 + 1e-5)
        metrics = optimizer.last_tucker_lr_scaling_metrics
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/base_lr"], 0.05)
        self.assertLessEqual(
            metrics["tucker_lr_scaling/layer/analytic_delta_bound"],
            0.05 + 1e-6,
        )

    def test_first_order_omits_higher_order_terms(self):
        sigmas = [2.0, 3.0, 5.0]
        rhos = [0.5, 0.25, 0.1]
        actual = tucker_spectral_denominator(sigmas, rhos, mode="first_order")
        expected = 0.5 * 3.0 * 5.0 + 0.25 * 2.0 * 5.0 + 0.1 * 2.0 * 3.0
        torch.testing.assert_close(actual, torch.tensor(expected))

    def test_scheduler_group_lr_is_used_once(self):
        core = torch.nn.Parameter(torch.randn(1, 1, dtype=torch.float64))
        factors = tuple(
            torch.nn.Parameter(torch.ones(1, 1, dtype=torch.float64))
            for _ in range(4)
        )
        core.grad = torch.ones_like(core)
        for factor in factors:
            factor.grad = torch.ones_like(factor)
        optimizer = TensorionOptimizer(
            tensorion_params=[("core", core, (1, 1, 1, 1))],
            riemannian_muon_params=[
                (f"U{index}", factor) for index, factor in enumerate(factors, start=1)
            ],
            adamw_param_groups=[],
            tucker_module_specs=[("layer", core, factors)],
            tucker_lr_scaling_mode="spectron_bound",
            tucker_lr_scaling_exact_svd_debug=True,
            tucker_lr_scaling_log_interval=1,
            lr=0.2,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )
        for group in optimizer.param_groups:
            group["lr"] = 0.0125
        optimizer.step()
        metrics = optimizer.last_tucker_lr_scaling_metrics
        self.assertAlmostEqual(metrics["tucker_lr_scaling/layer/base_lr"], 0.0125)
        self.assertAlmostEqual(
            metrics["tucker_lr_scaling/layer/alpha"],
            0.0125 / metrics["tucker_lr_scaling/layer/denominator"],
        )

    def test_calibrated_first_order_starts_at_legacy_coefficient(self):
        core = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float64))
        factors = tuple(
            torch.nn.Parameter(_orthogonal(rows, 2))
            for rows in (3, 4, 5, 6)
        )
        core.grad = torch.randn_like(core)
        for factor in factors:
            factor.grad = torch.randn_like(factor)
        optimizer = TensorionOptimizer(
            tensorion_params=[("core", core, (2, 2, 2, 2))],
            riemannian_muon_params=[
                (f"U{index}", factor)
                for index, factor in enumerate(factors, start=1)
            ],
            adamw_param_groups=[],
            tucker_module_specs=[("layer", core, factors)],
            tucker_lr_scaling_mode="first_order_calibrated",
            tucker_lr_scaling_exact_svd_debug=True,
            tucker_lr_scaling_log_interval=1,
            lr=0.05,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )

        optimizer.step()

        metrics = optimizer.last_tucker_lr_scaling_metrics
        self.assertAlmostEqual(
            metrics["tucker_lr_scaling/layer/alpha"],
            0.05,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["tucker_lr_scaling/layer/reference_denominator"],
            metrics["tucker_lr_scaling/layer/denominator"],
            places=6,
        )

    def test_functional_paper_mup_removes_shape_scale_and_matches_proxy(self):
        core = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float64))
        factors = tuple(
            torch.nn.Parameter(_orthogonal(rows, 2))
            for rows in (3, 4, 5, 6)
        )
        core.grad = torch.randn_like(core)
        for factor in factors:
            factor.grad = torch.randn_like(factor)
        optimizer = TensorionOptimizer(
            tensorion_params=[("core", core, (2, 2, 2, 2))],
            riemannian_muon_params=[
                (f"U{index}", factor)
                for index, factor in enumerate(factors, start=1)
            ],
            adamw_param_groups=[],
            tucker_module_specs=[("layer", core, factors)],
            tucker_lr_scaling_mode="paper_mup_functional",
            tucker_lr_scaling_exact_svd_debug=True,
            tucker_lr_scaling_log_interval=1,
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=True,
            orthogonalization="svd",
        )

        optimizer.step()

        metrics = optimizer.last_tucker_lr_scaling_metrics
        prefix = "tucker_lr_scaling/layer"
        normalizer = metrics[f"{prefix}/functional_normalizer"]
        self.assertGreater(normalizer, 0.0)
        self.assertAlmostEqual(
            metrics[f"{prefix}/legacy_functional_proxy"],
            metrics[f"{prefix}/raw_paper_functional_proxy"] * normalizer,
            places=5,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/applied_step_coefficient_core"],
            metrics[f"{prefix}/effective_lr_core"],
            places=12,
        )

    def test_power_vectors_round_trip_through_optimizer_checkpoint(self):
        parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
        optimizer = TensorionOptimizer(
            tensorion_params=[("matrix", parameter, parameter.shape)],
            adamw_param_groups=[],
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
        )
        state = optimizer.state[parameter]
        warm_started_spectral_norm(
            parameter,
            state,
            prefix="spectron_weight",
            power_iters=1,
        )
        saved = copy.deepcopy(optimizer.state_dict())

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = TensorionOptimizer(
            tensorion_params=[("matrix", restored_parameter, restored_parameter.shape)],
            adamw_param_groups=[],
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
        )
        restored.load_state_dict(saved)
        restored_state = restored.state[restored_parameter]
        torch.testing.assert_close(
            restored_state["spectron_weight_left"],
            state["spectron_weight_left"],
        )
        torch.testing.assert_close(
            restored_state["spectron_weight_right"],
            state["spectron_weight_right"],
        )

    def test_bfloat16_norm_uses_fp32_state(self):
        matrix = torch.randn(7, 5, dtype=torch.bfloat16)
        state = {}
        sigma = warm_started_spectral_norm(
            matrix,
            state,
            prefix="mixed_precision",
            power_iters=1,
        )
        self.assertEqual(sigma.dtype, torch.float32)
        self.assertEqual(state["mixed_precision_left"].dtype, torch.float32)
        self.assertEqual(state["mixed_precision_right"].dtype, torch.float32)

    def test_none_mode_reproduces_legacy_tensorion_step(self):
        initial = torch.randn(2, 3, dtype=torch.float64)
        grad = torch.randn_like(initial)
        first = torch.nn.Parameter(initial.clone())
        second = torch.nn.Parameter(initial.clone())
        first.grad = grad.clone()
        second.grad = grad.clone()
        common = dict(
            tensorion_params=[("weight", first, first.shape)],
            adamw_param_groups=[],
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )
        legacy = TensorionOptimizer(**common)
        legacy.step()

        compatible = TensorionOptimizer(
            tensorion_params=[("weight", second, second.shape)],
            adamw_param_groups=[],
            tucker_module_specs=[],
            tucker_lr_scaling_mode="none",
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )
        compatible.step()
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
