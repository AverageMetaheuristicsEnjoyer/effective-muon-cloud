import math
import unittest

import torch
import triton

from torchao.optim.quant_utils import create_dynamic_map

from third_party.solo.triton.adamw import solo_adamw_step


BLOCK_SIZE = 128
QMAP_SIGNED = create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)


def _dequant_de_4bit(codes, scale, qmap, block_size):
    """Dequantize DE 4-bit packed codes → float32."""
    unpacked = torch.stack([codes >> 4, codes & 0xF], dim=-1).view(-1)
    qmap_t = torch.tensor(qmap, device=codes.device)
    return (qmap_t[unpacked.long()].view(-1, block_size) * scale.view(-1, 1)).view(-1)


def _dequant_qema_2bit(codes, scale, alpha, block_size):
    """Dequantize qema 2-bit packed codes → float32."""
    unpacked = torch.stack([
        codes >> 6, (codes >> 4) & 0x3, (codes >> 2) & 0x3, codes & 0x3
    ], dim=-1).view(-1).float()
    return (
        (alpha.view(-1, 1) ** unpacked.view(-1, block_size)) * scale.view(-1, 1)
    ).view(-1)


def _make_state(n_elements, block_size, device="cuda"):
    """Create random initial quantized state + param/grad for testing."""
    n_blocks = n_elements // block_size

    param = torch.randn(n_elements, device=device) * 0.1
    grad = torch.randn(n_elements, device=device) * 0.01

    # 1st moment: DE 4-bit (random codes 0-15, packed)
    m1_codes_flat = torch.randint(0, 16, (n_elements,), device=device, dtype=torch.uint8)
    m1_codes = (m1_codes_flat[::2] << 4) | m1_codes_flat[1::2]
    m1_scale = torch.rand(n_blocks, device=device) * 0.1 + 0.01
    m1_qmap = torch.tensor(QMAP_SIGNED, device=device, dtype=torch.float32)

    # 2nd moment: qema 2-bit (random codes 0-3, packed)
    m2_codes_flat = torch.randint(0, 4, (n_elements,), device=device, dtype=torch.uint8)
    m2_codes = (m2_codes_flat[::4] << 6) | (m2_codes_flat[1::4] << 4) | (m2_codes_flat[2::4] << 2) | m2_codes_flat[3::4]
    m2_scale = torch.rand(n_blocks, device=device) * 0.1 + 0.01
    m2_alpha = torch.rand(n_blocks, device=device) * 0.3 + 0.5  # alpha in (0.5, 0.8)

    return param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha


class TestSoloAdamWStep(unittest.TestCase):
    """Test fused Triton AdamW step against PyTorch reference."""

    def _reference_step(self, param, grad, m1_float, m2_float,
                        lr, beta1, beta2, eps, weight_decay, step):
        """Pure float32 reference: returns updated param, new_m1, new_m2."""
        new_m1 = beta1 * m1_float + (1 - beta1) * grad
        new_m2 = beta2 * m2_float + (1 - beta2) * grad ** 2

        bias_correction1 = 1 - beta1 ** step
        bias_correction2_sqrt = math.sqrt(1 - beta2 ** step)
        step_size = lr / bias_correction1

        p = param.float() * (1 - lr * weight_decay)
        denom = new_m2.sqrt() / bias_correction2_sqrt + eps
        p = p - step_size * new_m1 / denom
        return p, new_m1, new_m2

    def test_param_update(self):
        """Param update should match float32 reference (same dequanted inputs)."""
        n = BLOCK_SIZE * 8
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        # Dequant to get float reference inputs
        m1_float = _dequant_de_4bit(m1_codes, m1_scale, QMAP_SIGNED, BLOCK_SIZE)
        m2_float = _dequant_qema_2bit(m2_codes, m2_scale, m2_alpha, BLOCK_SIZE)

        hp = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=5)
        ref_param, _, _ = self._reference_step(param, grad, m1_float, m2_float, **hp)

        param_triton = param.clone()
        solo_adamw_step(
            param_triton, grad,
            m1_codes.clone(), m1_scale.clone(), m1_qmap,
            m2_codes.clone(), m2_scale.clone(), m2_alpha.clone(),
            quantile=0.1, block_size=BLOCK_SIZE, seed=42, **hp,
        )

        torch.testing.assert_close(param_triton, ref_param, atol=1e-5, rtol=1e-5)

    def test_m1_scale(self):
        """1st moment scale should be abs-max of updated m1."""
        n = BLOCK_SIZE * 4
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        m1_float = _dequant_de_4bit(m1_codes, m1_scale, QMAP_SIGNED, BLOCK_SIZE)
        m2_float = _dequant_qema_2bit(m2_codes, m2_scale, m2_alpha, BLOCK_SIZE)

        hp = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1)
        _, new_m1, _ = self._reference_step(param, grad, m1_float, m2_float, **hp)
        expected_m1_scale = new_m1.view(-1, BLOCK_SIZE).abs().amax(-1).clamp(1e-12)

        m1_scale_out = m1_scale.clone()
        solo_adamw_step(
            param.clone(), grad,
            m1_codes.clone(), m1_scale_out, m1_qmap,
            m2_codes.clone(), m2_scale.clone(), m2_alpha.clone(),
            quantile=0.1, block_size=BLOCK_SIZE, seed=42, **hp,
        )

        torch.testing.assert_close(m1_scale_out, expected_m1_scale, atol=1e-5, rtol=1e-5)

    def test_m2_scale(self):
        """2nd moment scale should be max of updated m2."""
        n = BLOCK_SIZE * 4
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        m1_float = _dequant_de_4bit(m1_codes, m1_scale, QMAP_SIGNED, BLOCK_SIZE)
        m2_float = _dequant_qema_2bit(m2_codes, m2_scale, m2_alpha, BLOCK_SIZE)

        hp = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1)
        _, _, new_m2 = self._reference_step(param, grad, m1_float, m2_float, **hp)
        expected_m2_scale = new_m2.view(-1, BLOCK_SIZE).amax(-1).clamp(1e-12)

        m2_scale_out = m2_scale.clone()
        solo_adamw_step(
            param.clone(), grad,
            m1_codes.clone(), m1_scale.clone(), m1_qmap,
            m2_codes.clone(), m2_scale_out, m2_alpha.clone(),
            quantile=0.1, block_size=BLOCK_SIZE, seed=42, **hp,
        )

        torch.testing.assert_close(m2_scale_out, expected_m2_scale, atol=1e-5, rtol=1e-5)

    def test_m1_dequant_close(self):
        """Dequanted 1st moment after re-quantization should be close to float reference."""
        n = BLOCK_SIZE * 8
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        m1_float = _dequant_de_4bit(m1_codes, m1_scale, QMAP_SIGNED, BLOCK_SIZE)
        m2_float = _dequant_qema_2bit(m2_codes, m2_scale, m2_alpha, BLOCK_SIZE)

        hp = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1)
        _, ref_m1, _ = self._reference_step(param, grad, m1_float, m2_float, **hp)

        m1_codes_out = m1_codes.clone()
        m1_scale_out = m1_scale.clone()
        solo_adamw_step(
            param.clone(), grad,
            m1_codes_out, m1_scale_out, m1_qmap,
            m2_codes.clone(), m2_scale.clone(), m2_alpha.clone(),
            quantile=0.1, block_size=BLOCK_SIZE, seed=42, **hp,
        )

        m1_dequant = _dequant_de_4bit(m1_codes_out, m1_scale_out, QMAP_SIGNED, BLOCK_SIZE)
        # 4-bit quantization error: scale * max_qmap_step / 2
        torch.testing.assert_close(m1_dequant, ref_m1, atol=0.05, rtol=0.2)

    def test_m2_dequant_close(self):
        """Dequanted 2nd moment after re-quantization should be close to float reference."""
        n = BLOCK_SIZE * 8
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        m1_float = _dequant_de_4bit(m1_codes, m1_scale, QMAP_SIGNED, BLOCK_SIZE)
        m2_float = _dequant_qema_2bit(m2_codes, m2_scale, m2_alpha, BLOCK_SIZE)

        hp = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1)
        _, _, ref_m2 = self._reference_step(param, grad, m1_float, m2_float, **hp)

        m2_codes_out = m2_codes.clone()
        m2_scale_out = m2_scale.clone()
        m2_alpha_out = m2_alpha.clone()
        solo_adamw_step(
            param.clone(), grad,
            m1_codes.clone(), m1_scale.clone(), m1_qmap,
            m2_codes_out, m2_scale_out, m2_alpha_out,
            quantile=0.1, block_size=BLOCK_SIZE, seed=42, **hp,
        )

        m2_dequant = _dequant_qema_2bit(m2_codes_out, m2_scale_out, m2_alpha_out, BLOCK_SIZE)
        # 2-bit quantization is coarser
        torch.testing.assert_close(m2_dequant, ref_m2, atol=0.1, rtol=0.5)

    def test_bf16_param(self):
        """Should work with bf16 params and grads."""
        n = BLOCK_SIZE * 4
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        param_bf16 = param.bfloat16()
        grad_bf16 = grad.bfloat16()

        # Should not error
        solo_adamw_step(
            param_bf16, grad_bf16,
            m1_codes, m1_scale, m1_qmap,
            m2_codes, m2_scale, m2_alpha,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.01, step=1, quantile=0.1, block_size=BLOCK_SIZE,
        )
        self.assertFalse(param_bf16.isnan().any())
        self.assertFalse(param_bf16.isinf().any())

    def test_many_blocks(self):
        """Large tensor: 1000 blocks."""
        n = BLOCK_SIZE * 1000
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        solo_adamw_step(
            param, grad,
            m1_codes, m1_scale, m1_qmap,
            m2_codes, m2_scale, m2_alpha,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.01, step=10, quantile=0.1, block_size=BLOCK_SIZE,
        )
        self.assertFalse(param.isnan().any())


class TestSoloAdamWBenchmark(unittest.TestCase):
    """Speed benchmark (prints results, always passes)."""

    def test_benchmark(self):
        n = BLOCK_SIZE * 100_000  # ~12.8M elements
        param, grad, m1_codes, m1_scale, m1_qmap, m2_codes, m2_scale, m2_alpha = _make_state(n, BLOCK_SIZE)

        def triton_fn():
            solo_adamw_step(
                param, grad,
                m1_codes, m1_scale, m1_qmap,
                m2_codes, m2_scale, m2_alpha,
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                weight_decay=0.01, step=1, quantile=0.1, block_size=BLOCK_SIZE,
                seed=42,
            )

        ms = triton.testing.do_bench(triton_fn)
        throughput = n * 4 / ms / 1e6  # GB/s (float32 equiv)
        print(f"\nFused AdamW step: {ms:.3f} ms  ({throughput:.1f} GB/s for {n/1e6:.1f}M elements)")


if __name__ == "__main__":
    unittest.main()
