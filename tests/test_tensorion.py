import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.tucker_linear import TuckerLinear
from optim.tensorion import (
    TensorionOptimizer,
    fold_tensor,
    select_balanced_unfolding,
    stiefel_tangent_projection,
    tensorion_direction,
    tucker_core_shape_overrides,
    unfold_tensor,
)


class TensorionTest(unittest.TestCase):
    def test_stiefel_projection_matches_formula_and_is_tangent(self):
        torch.manual_seed(41)
        point, _ = torch.linalg.qr(
            torch.randn(7, 3, dtype=torch.float64),
            mode="reduced",
        )
        vector = torch.randn_like(point)

        projected = stiefel_tangent_projection(point, vector)
        cross = point.mT @ vector
        expected = vector - point @ (0.5 * (cross + cross.mT))
        tangency_violation = point.mT @ projected + projected.mT @ point

        torch.testing.assert_close(projected, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            tangency_violation,
            torch.zeros_like(tangency_violation),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_riemannian_muon_projects_factor_gradient_before_momentum(self):
        point = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                dtype=torch.float64,
            )
        )
        # This gradient is purely normal: G = X S with symmetric S.
        point.grad = torch.tensor(
            [[2.0, -0.5], [-0.5, 3.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        before = point.detach().clone()
        optimizer = TensorionOptimizer(
            tensorion_params=[],
            muon_params=[],
            riemannian_muon_params=[("factor", point)],
            adamw_param_groups=[],
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="ns",
            ns_steps=6,
        )

        optimizer.step()

        torch.testing.assert_close(point, before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            optimizer.state[point]["momentum_buffer"],
            torch.zeros_like(point),
            rtol=0.0,
            atol=0.0,
        )

    def test_offline_unfolding_maximizes_balanced_dimension(self):
        shape = (2, 3, 4, 5)
        plan = select_balanced_unfolding(shape)

        scores = []
        full_mask = (1 << len(shape)) - 1
        for mask in range(1, full_mask):
            rows = math.prod(
                shape[index] for index in range(len(shape)) if mask & (1 << index)
            )
            columns = math.prod(shape) // rows
            scores.append(min(rows, columns))

        self.assertEqual(min(plan.rows, plan.columns), max(scores))
        self.assertEqual(plan.rows * plan.columns, math.prod(shape))

    def test_fold_is_inverse_of_unfold(self):
        tensor = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        plan = select_balanced_unfolding(tensor.shape)
        restored = fold_tensor(unfold_tensor(tensor, plan), plan)
        torch.testing.assert_close(restored, tensor)

    def test_matrix_case_recovers_muon_lmo_with_exact_svd(self):
        momentum = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]],
            dtype=torch.float64,
        )
        plan = select_balanced_unfolding(momentum.shape)
        update = tensorion_direction(
            momentum,
            plan,
            orthogonalization="svd",
        )

        unfolded = unfold_tensor(momentum, plan)
        left, _, right_t = torch.linalg.svd(unfolded, full_matrices=False)
        expected = fold_tensor(left @ right_t, plan)
        torch.testing.assert_close(update, expected, rtol=1e-12, atol=1e-12)

    def test_optimizer_step_matches_algorithm_one_without_lr_adjustment(self):
        parameter = torch.nn.Parameter(
            torch.tensor(
                [[0.2, -0.1, 0.4], [0.7, -0.3, 0.5]],
                dtype=torch.float64,
            )
        )
        grad = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]],
            dtype=torch.float64,
        )
        before = parameter.detach().clone()
        parameter.grad = grad.clone()
        optimizer = TensorionOptimizer(
            tensorion_params=[("weight", parameter, parameter.shape)],
            adamw_param_groups=[],
            lr=0.1,
            weight_decay=0.0,
            momentum=0.0,
            adjust_lr=False,
            orthogonalization="svd",
        )

        optimizer.step()

        plan = select_balanced_unfolding(parameter.shape)
        expected_update = tensorion_direction(
            grad,
            plan,
            orthogonalization="svd",
        )
        torch.testing.assert_close(
            parameter,
            before - 0.1 * expected_update,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_adamw_fallback_matches_torch_adamw(self):
        ours_param = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
        torch_param = torch.nn.Parameter(ours_param.detach().clone())
        grad = torch.tensor([0.3, -0.1], dtype=torch.float64)
        ours_param.grad = grad.clone()
        torch_param.grad = grad.clone()

        ours = TensorionOptimizer(
            tensorion_params=[],
            adamw_param_groups=[{"params": [ours_param], "weight_decay": 0.1}],
            lr=1e-3,
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
        )
        reference = torch.optim.AdamW(
            [torch_param],
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )

        ours.step()
        reference.step()
        torch.testing.assert_close(ours_param, torch_param, rtol=1e-12, atol=1e-12)

    def test_muon_matrix_group_uses_nesterov_orthogonalized_update(self):
        parameter = torch.nn.Parameter(
            torch.tensor(
                [[0.2, -0.1, 0.4], [0.7, -0.3, 0.5]],
                dtype=torch.float64,
            )
        )
        grad = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]],
            dtype=torch.float64,
        )
        before = parameter.detach().clone()
        parameter.grad = grad.clone()
        optimizer = TensorionOptimizer(
            tensorion_params=[],
            muon_params=[("factor", parameter)],
            adamw_param_groups=[],
            lr=0.1,
            weight_decay=0.0,
            momentum=0.95,
            adjust_lr=False,
            orthogonalization="svd",
        )

        optimizer.step()

        expected_update = tensorion_direction(
            grad,
            select_balanced_unfolding(parameter.shape),
            orthogonalization="svd",
        )
        torch.testing.assert_close(
            parameter,
            before - 0.1 * expected_update,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_tucker_core_uses_logical_four_dimensional_shape(self):
        module = TuckerLinear(
            12,
            18,
            rank=2,
            bias=False,
            equal_params=False,
            dtype=torch.float64,
        )
        overrides = tucker_core_shape_overrides(module)
        r1, r2, r3, r4 = module.ranks
        self.assertEqual(overrides[module.core_matrix], (r3, r4, r1, r2))
        self.assertEqual(
            math.prod(overrides[module.core_matrix]),
            module.core_matrix.numel(),
        )


if __name__ == "__main__":
    unittest.main()
