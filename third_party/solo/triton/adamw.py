import math

import torch
from torch.optim import Optimizer
import triton
import triton.language as tl

from torchao.optim.quant_utils import create_dynamic_map

QMAP_SIGNED_DE = create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)


@triton.jit
def solo_adamw_step_kernel(
    # Param + grad
    param_ptr,
    grad_ptr,
    # 1st moment (DE 4-bit signed)
    m1_codes_ptr,
    m1_scale_ptr,
    m1_qmap_ptr,
    # 2nd moment (qema 2-bit unsigned)
    m2_codes_ptr,
    m2_scale_ptr,
    m2_alpha_ptr,
    # Adam hyperparams (precomputed where possible)
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step_size,              # lr / (1 - beta1^step)
    bias_correction2_sqrt,  # sqrt(1 - beta2^step)
    # Meta
    n_elements,
    seed,
    # Constexpr
    p: tl.constexpr,         # quantile for qema
    NUMS: tl.constexpr,      # 2^bits - 1 = 3 for 2-bit
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    elem_offsets = pid * BLOCK_SIZE + offsets
    mask = elem_offsets < n_elements

    # === Load param and grad ===
    param = tl.load(param_ptr + elem_offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + elem_offsets, mask=mask).to(tl.float32)

    # === Dequant 1st moment (DE 4-bit signed) ===
    # Packing: byte[k] = (code[2k] << 4) | code[2k+1]
    m1_byte_offsets = pid * (BLOCK_SIZE // 2) + offsets // 2
    m1_packed = tl.load(m1_codes_ptr + m1_byte_offsets)
    m1_shift = (1 - offsets % 2) * 4  # even→4, odd→0
    m1_codes = (m1_packed >> m1_shift) & 0xF

    m1_scale = tl.load(m1_scale_ptr + pid)
    m1_val = tl.load(m1_qmap_ptr + m1_codes) * m1_scale

    # === Dequant 2nd moment (qema 2-bit unsigned) ===
    # Packing: byte[k] = (c[4k]<<6)|(c[4k+1]<<4)|(c[4k+2]<<2)|c[4k+3]
    m2_byte_offsets = pid * (BLOCK_SIZE // 4) + offsets // 4
    m2_packed = tl.load(m2_codes_ptr + m2_byte_offsets)
    m2_shift = (3 - offsets % 4) * 2
    m2_codes = (m2_packed >> m2_shift) & 0x3

    m2_scale = tl.load(m2_scale_ptr + pid)
    m2_alpha = tl.load(m2_alpha_ptr + pid)
    log2_alpha_old = tl.math.log2(m2_alpha)
    m2_val = tl.math.exp2(m2_codes.to(tl.float32) * log2_alpha_old) * m2_scale

    # === EMA updates ===
    new_m1 = beta1 * m1_val + (1.0 - beta1) * grad
    new_m2 = beta2 * m2_val + (1.0 - beta2) * grad * grad

    # === Adam param update ===
    param = param * (1.0 - lr * weight_decay)
    denom = tl.sqrt(new_m2) / bias_correction2_sqrt + eps
    param = param - step_size * new_m1 / denom
    tl.store(param_ptr + elem_offsets, param, mask=mask)

    # === Quantize 1st moment (DE 4-bit, deterministic rounding) ===
    new_m1_scale = tl.max(tl.abs(new_m1), axis=0)
    new_m1_scale = tl.maximum(new_m1_scale, 1e-12)
    m1_norm = new_m1 / new_m1_scale

    # Branchless binary search in qmap[0..15]
    new_m1_codes = tl.where(m1_norm >= tl.load(m1_qmap_ptr + 8), 8, 0)
    new_m1_codes += tl.where(m1_norm >= tl.load(m1_qmap_ptr + new_m1_codes + 4), 4, 0)
    new_m1_codes += tl.where(m1_norm >= tl.load(m1_qmap_ptr + new_m1_codes + 2), 2, 0)
    new_m1_codes += tl.where(m1_norm >= tl.load(m1_qmap_ptr + new_m1_codes + 1), 1, 0)

    # Deterministic rounding (nearest)
    new_m1_codes = tl.minimum(new_m1_codes, 15)
    codes_up_m1 = tl.minimum(new_m1_codes + 1, 15)
    val_down_m1 = tl.load(m1_qmap_ptr + new_m1_codes)
    val_up_m1 = tl.load(m1_qmap_ptr + codes_up_m1)
    residual_m1 = m1_norm - val_down_m1
    new_m1_codes = tl.where(residual_m1 >= (val_up_m1 - val_down_m1) * 0.5, codes_up_m1, new_m1_codes)
    new_m1_codes = new_m1_codes.to(tl.uint8)

    # Pack 4-bit
    m1_shifted = new_m1_codes << m1_shift
    m1_packed_new = tl.reshape(m1_shifted, (BLOCK_SIZE // 2, 2))
    m1_packed_new = tl.sum(m1_packed_new, axis=1).to(tl.uint8)

    m1_store_offsets = pid * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    tl.store(m1_codes_ptr + m1_store_offsets, m1_packed_new)
    tl.store(m1_scale_ptr + pid, new_m1_scale)

    # === Quantize 2nd moment (qema 2-bit, stochastic rounding) ===
    new_m2_scale = tl.max(new_m2, axis=0)
    new_m2_scale = tl.maximum(new_m2_scale, 1e-12)

    # Quantile via sort for alpha
    sorted_m2 = tl.sort(new_m2)
    idx_lo: tl.constexpr = int(p * (BLOCK_SIZE - 1))
    idx_hi: tl.constexpr = idx_lo + 1 if idx_lo < BLOCK_SIZE - 1 else idx_lo
    frac: tl.constexpr = p * (BLOCK_SIZE - 1) - idx_lo
    q_lo = tl.sum(tl.where(offsets == idx_lo, sorted_m2, 0.0))
    q_hi = tl.sum(tl.where(offsets == idx_hi, sorted_m2, 0.0))
    quantile_val = q_lo + frac * (q_hi - q_lo)

    xp = quantile_val / new_m2_scale
    xp = tl.maximum(xp, 1e-12)
    new_log2_alpha = tl.math.log2(xp) / NUMS
    new_m2_alpha = tl.math.exp2(new_log2_alpha)

    # Logarithmic quantization
    m2_norm = new_m2 / new_m2_scale
    rand = tl.rand(seed, pid * BLOCK_SIZE + offsets)
    new_m2_codes = tl.extra.cuda.libdevice.round(tl.math.log2(m2_norm) / new_log2_alpha + rand - 0.5)
    new_m2_codes = tl.minimum(tl.maximum(new_m2_codes, 0), 3).to(tl.uint8)

    # Pack 2-bit
    m2_shifted = new_m2_codes << m2_shift
    m2_packed_new = tl.reshape(m2_shifted, (BLOCK_SIZE // 4, 4))
    m2_packed_new = tl.sum(m2_packed_new, axis=1).to(tl.uint8)

    m2_store_offsets = pid * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
    tl.store(m2_codes_ptr + m2_store_offsets, m2_packed_new)
    tl.store(m2_scale_ptr + pid, new_m2_scale)
    tl.store(m2_alpha_ptr + pid, new_m2_alpha)


def solo_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    m1_codes: torch.Tensor,
    m1_scale: torch.Tensor,
    m1_qmap: torch.Tensor,
    m2_codes: torch.Tensor,
    m2_scale: torch.Tensor,
    m2_alpha: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    quantile: float,
    block_size: int,
    seed: int | None = None,
):
    n_elements = param.numel()
    assert n_elements % block_size == 0

    n_blocks = n_elements // block_size
    bias_correction1 = 1 - beta1 ** step
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(1 - beta2 ** step)

    if seed is None:
        seed = torch.randint(2**31, size=(), device=param.device).item()

    NUMS = 3  # 2^2 - 1 for 2-bit

    solo_adamw_step_kernel[(n_blocks,)](
        param.view(-1), grad.view(-1),
        m1_codes, m1_scale, m1_qmap,
        m2_codes, m2_scale, m2_alpha,
        lr, beta1, beta2, eps, weight_decay,
        step_size, bias_correction2_sqrt,
        n_elements, seed,
        p=quantile, NUMS=NUMS, BLOCK_SIZE=block_size,
    )


class TritonSoloAdamW(Optimizer):
    """Fused 4/2-bit AdamW via a single Triton kernel per parameter.

    1st moment: 4-bit Dynamic Exponent (signed)
    2nd moment: 2-bit QEMA logarithmic (unsigned)

    Small params (numel < block_size or not divisible) fall back to float32 Adam.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, quantile=0.1, block_size=128):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.quantile = quantile
        self.block_size = block_size

    def _can_quantize(self, p):
        n = p.numel()
        return n >= self.block_size and n % self.block_size == 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    n = p.numel()
                    if self._can_quantize(p):
                        n_blocks = n // self.block_size
                        state['quantized'] = True
                        state['m1_codes'] = torch.zeros(n // 2, dtype=torch.uint8, device=p.device)
                        state['m1_scale'] = torch.zeros(n_blocks, dtype=torch.float32, device=p.device)
                        state['m1_qmap'] = torch.tensor(QMAP_SIGNED_DE, dtype=torch.float32, device=p.device)
                        state['m2_codes'] = torch.zeros(n // 4, dtype=torch.uint8, device=p.device)
                        state['m2_scale'] = torch.zeros(n_blocks, dtype=torch.float32, device=p.device)
                        state['m2_alpha'] = torch.ones(n_blocks, dtype=torch.float32, device=p.device)
                    else:
                        state['quantized'] = False
                        state['exp_avg'] = torch.zeros(n, dtype=torch.float32, device=p.device)
                        state['exp_avg_sq'] = torch.zeros(n, dtype=torch.float32, device=p.device)

                state['step'] += 1
                step = state['step']

                if state['quantized']:
                    solo_adamw_step(
                        p, p.grad,
                        state['m1_codes'], state['m1_scale'], state['m1_qmap'],
                        state['m2_codes'], state['m2_scale'], state['m2_alpha'],
                        lr=lr, beta1=beta1, beta2=beta2,
                        eps=eps, weight_decay=wd,
                        step=step, quantile=self.quantile,
                        block_size=self.block_size,
                    )
                else:
                    # Float32 fallback for small/incompatible params
                    grad = p.grad.float()
                    p_f32 = p.float()
                    p_f32.mul_(1 - lr * wd)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                    bc1 = 1 - beta1 ** step
                    bc2_sqrt = math.sqrt(1 - beta2 ** step)
                    denom = exp_avg_sq.sqrt().div_(bc2_sqrt).add_(eps)
                    p_f32.addcdiv_(exp_avg, denom, value=-lr / bc1)
                    p.copy_(p_f32)

        return loss
