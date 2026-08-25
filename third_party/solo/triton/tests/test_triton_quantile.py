import unittest

import torch
import triton

from third_party.solo.triton.qema_2bit import blockwise_quantile


class TestBlockwiseQuantile(unittest.TestCase):
    """Accuracy tests: blockwise_quantile vs torch.Tensor.quantile."""

    def _reference(self, data, block_size, p):
        return data.view(-1, block_size).quantile(p, dim=-1)

    def _check(self, data, block_size, p, atol=1e-5, rtol=1e-5):
        result = blockwise_quantile(data, block_size, p)
        expected = self._reference(data, block_size, p)
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)

    # --- Various (block_size, p) combos ---

    def test_128_p01(self):
        data = torch.rand(1024, device="cuda").abs()
        self._check(data, 128, 0.1)

    def test_128_p025(self):
        data = torch.rand(1024, device="cuda").abs()
        self._check(data, 128, 0.25)

    def test_128_p05(self):
        data = torch.rand(1024, device="cuda").abs()
        self._check(data, 128, 0.5)

    def test_64_p01(self):
        data = torch.rand(1024, device="cuda").abs()
        self._check(data, 64, 0.1)

    def test_256_p01(self):
        data = torch.rand(1024, device="cuda").abs()
        self._check(data, 256, 0.1)

    # --- Edge cases ---

    def test_p0_min(self):
        data = torch.rand(128, device="cuda").abs()
        self._check(data, 128, 0.0)

    def test_p1_max(self):
        data = torch.rand(128, device="cuda").abs()
        self._check(data, 128, 1.0)

    def test_uniform_blocks(self):
        data = torch.ones(256, device="cuda") * 3.14
        self._check(data, 128, 0.5)

    def test_single_block(self):
        data = torch.rand(128, device="cuda").abs()
        self._check(data, 128, 0.1)

    def test_many_blocks(self):
        data = torch.rand(128 * 10000, device="cuda").abs()
        self._check(data, 128, 0.1)

    def test_sorted_input(self):
        data = torch.linspace(0.01, 1.0, 256, device="cuda")
        self._check(data, 128, 0.25)

    def test_mixed_positive(self):
        data = torch.cat([
            torch.rand(64, device="cuda") * 0.01,
            torch.rand(64, device="cuda") * 100.0,
        ])
        self._check(data, 128, 0.1)


class TestBlockwiseQuantileBenchmark(unittest.TestCase):
    """Speed benchmark (prints results, always passes)."""

    def test_benchmark(self):
        n_blocks = 100_000
        block_size = 128
        data = torch.rand(n_blocks * block_size, device="cuda")

        def triton_fn():
            return blockwise_quantile(data, block_size, 0.1)

        def torch_fn():
            return data.view(-1, block_size).quantile(0.1, dim=-1)

        triton_ms = triton.testing.do_bench(triton_fn)
        torch_ms = triton.testing.do_bench(torch_fn)

        print(f"\n{'Triton':>10}: {triton_ms:.3f} ms")
        print(f"{'PyTorch':>10}: {torch_ms:.3f} ms")
        print(f"{'Speedup':>10}: {torch_ms / triton_ms:.2f}x")


if __name__ == "__main__":
    unittest.main()
