import torch
import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config({"BM": 32, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
    triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
    triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
], key=["M", "D", "K"])
@triton.jit
def _gate_up_forward(X, W, GU, OUT, M: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        x = tl.load(X + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0)
        wg = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < D) & (kk[:, None] < K), 0)
        wu = tl.load(W + (D + n[None, :]) * K + kk[:, None], (n[None, :] < D) & (kk[:, None] < K), 0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    dtype = GU.dtype.element_ty
    gate = gate.to(dtype)
    up = up.to(dtype)
    mask = (m[:, None] < M) & (n[None, :] < D)
    tl.store(GU + m[:, None] * (2 * D) + n[None, :], gate, mask)
    tl.store(GU + m[:, None] * (2 * D) + D + n[None, :], up, mask)
    gf = gate.to(tl.float32)
    silu = (gf * tl.sigmoid(gf)).to(dtype)
    tl.store(OUT + m[:, None] * D + n[None, :], silu * up, mask)


@triton.jit
def _packed_swiglu_forward(GU, OUT, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // D, idx % D
    gate = tl.load(GU + row * (2 * D) + col, idx < N, 0).to(tl.float32)
    up = tl.load(GU + row * (2 * D) + D + col, idx < N, 0)
    silu = (gate * tl.sigmoid(gate)).to(up.dtype)
    tl.store(OUT + idx, silu * up, idx < N)


@triton.jit
def _packed_swiglu_backward(GU, DY, DGU, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // D, idx % D
    dtype = GU.dtype.element_ty
    gate = tl.load(GU + row * (2 * D) + col, idx < N, 0).to(tl.float32)
    up = tl.load(GU + row * (2 * D) + D + col, idx < N, 0).to(tl.float32)
    dy = tl.load(DY + idx, idx < N, 0).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = (gate * sig).to(dtype).to(tl.float32)
    dg = (dy * up).to(dtype).to(tl.float32) * (sig * (1 + gate * (1 - sig)))
    du = dy * silu
    tl.store(DGU + row * (2 * D) + col, dg, idx < N)
    tl.store(DGU + row * (2 * D) + D + col, du, idx < N)


def _swiglu_gradient(gu, grad):
    grad = grad.contiguous()
    dgu = torch.empty_like(gu)
    d = gu.shape[-1] // 2
    _packed_swiglu_backward[(triton.cdiv(grad.numel(), 256),)](
        gu, grad, dgu, grad.numel(), d, 256,
    )
    return dgu


class _GateUpSwiGLU(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        weight = weight.contiguous()
        m, k = x.shape
        d = weight.shape[0] // 2
        gu = torch.empty((m, 2 * d), device=x.device, dtype=x.dtype)
        out = torch.empty((m, d), device=x.device, dtype=x.dtype)
        _gate_up_forward[lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(d, meta["BN"]))](
            x, weight, gu, out, m, d, k,
        )
        ctx.save_for_backward(x, weight, gu)
        ctx.input_shape = shape
        return out.view(*shape[:-1], d)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        x, weight, gu = ctx.saved_tensors
        dgu = _swiglu_gradient(gu, grad)
        dx = (dgu @ weight).view(ctx.input_shape)
        dw = dgu.T @ x
        return dx, dw


class _PackedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        gu = gu.contiguous()
        d = gu.shape[-1] // 2
        out = torch.empty((*gu.shape[:-1], d), device=gu.device, dtype=gu.dtype)
        _packed_swiglu_forward[(triton.cdiv(out.numel(), 256),)](gu, out, out.numel(), d, 256)
        ctx.save_for_backward(gu)
        return out

    @staticmethod
    def backward(ctx, grad):
        (gu,) = ctx.saved_tensors
        return _swiglu_gradient(gu, grad)


gate_up_swiglu = _GateUpSwiGLU.apply
packed_swiglu = _PackedSwiGLU.apply
