import torch
import triton
import triton.language as tl


# --- Standalone blockwise quantile ---

@triton.jit
def blockwise_quantile_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    ptrs = input_ptr + pid * BLOCK_SIZE + offsets
    mask = pid * BLOCK_SIZE + offsets < n_elements
    block = tl.load(ptrs, mask=mask, other=0.0)

    sorted_block = tl.sort(block)

    # Linear interpolation indices (all constexpr)
    idx_lo: tl.constexpr = int(p * (BLOCK_SIZE - 1))
    idx_hi: tl.constexpr = idx_lo + 1 if idx_lo < BLOCK_SIZE - 1 else idx_lo
    frac: tl.constexpr = p * (BLOCK_SIZE - 1) - idx_lo

    # Extract values at idx_lo and idx_hi via masked reduction
    val_lo = tl.sum(tl.where(offsets == idx_lo, sorted_block, 0.0))
    val_hi = tl.sum(tl.where(offsets == idx_hi, sorted_block, 0.0))

    result = val_lo + frac * (val_hi - val_lo)
    tl.store(output_ptr + pid, result)


def blockwise_quantile(input: torch.Tensor, block_size: int, p: float) -> torch.Tensor:
    assert input.ndim == 1, "input must be 1D"
    assert input.numel() % block_size == 0, "input length must be divisible by block_size"
    assert block_size > 0 and (block_size & (block_size - 1)) == 0, "block_size must be a power of 2"

    if input.dtype != torch.float32:
        input = input.float()

    n_elements = input.numel()
    n_blocks = n_elements // block_size
    output = torch.empty(n_blocks, device=input.device, dtype=torch.float32)

    blockwise_quantile_kernel[(n_blocks,)](
        input, output, n_elements,
        p=p, BLOCK_SIZE=block_size,
    )
    return output


# --- Fused qema update with in-kernel quantile for alpha ---

@triton.jit
def qema_update_2bit_kernel(
    codes_ptr,
    scale_ptr,
    alpha_ptr,
    signal_ptr,
    beta,
    n_elements,
    seed,
    p: tl.constexpr,
    NUMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # --- Dequantize ---
    offsets = tl.arange(0, BLOCK_SIZE)
    byte_offsets = pid * (BLOCK_SIZE // 4) + offsets // 4
    packed = tl.load(codes_ptr + byte_offsets)
    shift = (3 - (offsets % 4)) * 2
    old_codes = (packed >> shift) & 0x3

    scale = tl.load(scale_ptr + pid)
    alpha = tl.load(alpha_ptr + pid)

    log2_alpha = tl.math.log2(alpha)
    old_val = tl.math.exp2(old_codes.to(tl.float32) * log2_alpha) * scale

    # --- EMA update ---
    signal = tl.load(signal_ptr + pid * BLOCK_SIZE + offsets)
    new_val = beta * old_val + (1 - beta) * signal

    # --- Scale + quantile via sort ---
    new_scale = tl.max(new_val, axis=0)
    new_scale = tl.maximum(new_scale, 1e-12)

    sorted_val = tl.sort(new_val)

    idx_lo: tl.constexpr = int(p * (BLOCK_SIZE - 1))
    idx_hi: tl.constexpr = idx_lo + 1 if idx_lo < BLOCK_SIZE - 1 else idx_lo
    frac: tl.constexpr = p * (BLOCK_SIZE - 1) - idx_lo

    val_lo = tl.sum(tl.where(offsets == idx_lo, sorted_val, 0.0))
    val_hi = tl.sum(tl.where(offsets == idx_hi, sorted_val, 0.0))
    quantile_val = val_lo + frac * (val_hi - val_lo)

    # alpha = (quantile / scale) ^ (1 / NUMS)
    xp = quantile_val / new_scale
    xp = tl.maximum(xp, 1e-12)
    log2_alpha = tl.math.log2(xp) / NUMS
    new_alpha = tl.math.exp2(log2_alpha)

    # --- Quantize ---
    normalized = new_val / new_scale
    rand = tl.rand(seed, pid * BLOCK_SIZE + offsets)
    new_codes = tl.extra.cuda.libdevice.round(tl.math.log2(normalized) / log2_alpha + rand - 0.5)
    new_codes = tl.minimum(tl.maximum(new_codes, 0), 3).to(tl.uint8)

    # --- Pack 4 x 2-bit codes into uint8 ---
    shifted = new_codes << shift
    packed = tl.reshape(shifted, (BLOCK_SIZE // 4, 4))
    packed = tl.sum(packed, axis=1)

    # --- Store ---
    byte_offsets = pid * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
    tl.store(codes_ptr + byte_offsets, packed)
    tl.store(scale_ptr + pid, new_scale)
    tl.store(alpha_ptr + pid, new_alpha)
