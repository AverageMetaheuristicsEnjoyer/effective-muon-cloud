import unittest

import torch
import torch.nn.functional as F

from models.tucker_chunked import (
    chunked_tucker_cross_entropy,
    chunked_tucker_linear,
)
from models.tucker_linear import TuckerLinear


class ChunkedTuckerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = TuckerLinear(
            12,
            18,
            rank=(2, 3, 2, 4),
            bias=False,
            equal_params=False,
            forward_mode="contract",
        ).to(self.device)

    def _parameter_grads(self):
        return [
            parameter.grad.detach().clone()
            for parameter in (
                self.module.core_matrix,
                self.module.U1,
                self.module.U2,
                self.module.U3,
                self.module.U4,
            )
        ]

    def test_linear_forward_and_backward_match_materialized_weight(self):
        x = torch.randn(2, 5, 12, device=self.device, requires_grad=True)
        upstream = torch.randn(2, 5, 18, device=self.device)
        dense = F.linear(x, self.module.materialize_weight(dtype=x.dtype))
        dense.backward(upstream)
        expected_x_grad = x.grad.detach().clone()
        expected_parameter_grads = self._parameter_grads()

        self.module.zero_grad(set_to_none=True)
        x.grad = None
        actual = chunked_tucker_linear(x, self.module, chunk_size=3)
        actual.backward(upstream)

        torch.testing.assert_close(actual, dense.detach(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(x.grad, expected_x_grad, rtol=3e-5, atol=3e-6)
        for actual_grad, expected_grad in zip(
            self._parameter_grads(), expected_parameter_grads
        ):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=4e-5, atol=4e-6
            )

    def test_fused_cross_entropy_matches_materialized_weight(self):
        x = torch.randn(3, 4, 12, device=self.device, requires_grad=True)
        targets = torch.randint(0, 18, (3, 4), device=self.device)
        targets[0, 0] = -1
        weight = self.module.materialize_weight(dtype=x.dtype)
        expected = F.cross_entropy(
            F.linear(x, weight).flatten(0, 1),
            targets.flatten(),
            ignore_index=-1,
            label_smoothing=0.1,
        )
        expected.backward()
        expected_x_grad = x.grad.detach().clone()
        expected_parameter_grads = self._parameter_grads()

        self.module.zero_grad(set_to_none=True)
        x.grad = None
        actual = chunked_tucker_cross_entropy(
            x,
            targets,
            self.module,
            chunk_size=5,
            ignore_index=-1,
            label_smoothing=0.1,
        )
        actual.backward()

        torch.testing.assert_close(actual, expected.detach(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(x.grad, expected_x_grad, rtol=4e-5, atol=4e-6)
        for actual_grad, expected_grad in zip(
            self._parameter_grads(), expected_parameter_grads
        ):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=5e-5, atol=5e-6
            )


if __name__ == "__main__":
    unittest.main()
