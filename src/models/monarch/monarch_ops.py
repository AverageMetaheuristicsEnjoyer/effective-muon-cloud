"""Opaque custom-op wrapper for the Monarch butterfly multiply.

Why: torch.compile/Inductor decomposes the eager butterfly's permute chain
into worse copies and generates a slow backward (measured +48% on fwd+bwd).
Registering the butterfly as a custom op makes it an atomic node: Inductor
fuses the surrounding glue (RMSNorm/RoPE/SwiGLU/residual) and leaves the
butterfly alone.

Also hosts the optional TileLang fused forward (monarch_tl.py); backward
stays on the cuBLAS bmm path in either case.

Usage:
    import monarch_ops
    monarch_ops.patch_monarch_linear(use_tilelang=False)   # opaque op only
    monarch_ops.patch_monarch_linear(use_tilelang=True)    # + TL fused fwd
"""
from __future__ import annotations

import numpy as np
import torch

from . import monarch_linear as _ml

_USE_TILELANG = False
_TL = None  # lazy module handle for monarch_tl


def _butterfly_fwd_math(x, w1_bfly, w2_bfly):
    """Eager forward math (same as BlockdiagButterflyMultiply.forward).

    Returns (out2, out1) with out1 saved for backward.
    """
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = int(np.prod(batch_shape))
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
    # (l, B, r) CONTIGUOUS (not a strided view) so the fake kernel matches
    out1 = (out1.transpose(0, 1).reshape(batch_dim, r, l)
            .permute(2, 0, 1).contiguous())
    out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    return out2, out1


def _butterfly_bwd_math(x, w1_bfly, w2_bfly, out1, dout):
    """Eager backward math (same as BlockdiagButterflyMultiply.backward)."""
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = int(np.prod(batch_shape))
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
    dout_reshaped = dout_reshaped.transpose(0, 1)
    dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
    dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
    dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
    dout1 = (dout1.transpose(0, 1).transpose(-1, -2).contiguous()
             .reshape(batch_dim, k, q).transpose(0, 1))
    dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
    dx = (torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1))
          .transpose(0, 1).reshape(*batch_shape, n))
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
    return dx, dw1_bfly, dw2_bfly


@torch.library.custom_op("monarch::butterfly", mutates_args=())
def _butterfly_op(x: torch.Tensor, w1_bfly: torch.Tensor,
                  w2_bfly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _USE_TILELANG:
        out2, out1 = _TL.tl_butterfly_fwd_v2(x, w1_bfly, w2_bfly)
        return out2, out1
    return _butterfly_fwd_math(x, w1_bfly, w2_bfly)


@_butterfly_op.register_fake
def _(x, w1_bfly, w2_bfly):
    batch_shape = x.shape[:-1]
    batch_dim = int(np.prod(batch_shape))
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    out2 = x.new_empty(*batch_shape, s * l)
    out1 = x.new_empty(l, batch_dim, r)
    return out2, out1


@torch.library.custom_op("monarch::butterfly_bwd", mutates_args=())
def _butterfly_bwd_op(x: torch.Tensor, w1_bfly: torch.Tensor,
                      w2_bfly: torch.Tensor, out1: torch.Tensor,
                      dout: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _butterfly_bwd_math(x, w1_bfly, w2_bfly, out1, dout)


@_butterfly_bwd_op.register_fake
def _(x, w1_bfly, w2_bfly, out1, dout):
    return (x.new_empty(x.shape), w1_bfly.new_empty(w1_bfly.shape),
            w2_bfly.new_empty(w2_bfly.shape))


def _setup_context(ctx, inputs, output):
    x, w1_bfly, w2_bfly = inputs
    ctx.save_for_backward(x, w1_bfly, w2_bfly, output[1])


def _backward(ctx, dout2, dout1_unused):
    x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
    dx, dw1, dw2 = _butterfly_bwd_op(x, w1_bfly, w2_bfly, out1,
                                     dout2.contiguous())
    return dx, dw1, dw2


_butterfly_op.register_autograd(_backward, setup_context=_setup_context)


def butterfly(x, w1_bfly, w2_bfly):
    """Drop-in replacement for blockdiag_butterfly_multiply (autocast-aware)."""
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
        x, w1_bfly, w2_bfly = x.to(dt), w1_bfly.to(dt), w2_bfly.to(dt)
    out2, _ = _butterfly_op(x, w1_bfly, w2_bfly)
    return out2


# ---------------------------------------------------------------------------
# v2: blocked-output ops. The op returns out2 in its NATURAL bmm layout
# (B, l, s) — no interleave copy inside. The feature-order permute happens in
# the python wrapper (compiled region), where Inductor fuses it into pointwise
# consumers; the backward receives a blocked grad, so the dout gather-copy
# becomes a strided VIEW for the cuBLAS bmms.
# ---------------------------------------------------------------------------

_USE_TRITON = False
_USE_FAST_RIFFLE = False


def _butterfly_fwd_blocked(x2, w1_bfly, w2_bfly):
    """x2 (B, n) -> out_blocked (B, l, s) contiguous, out1 (l, B, r) contiguous."""
    B = x2.shape[0]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    if _USE_TRITON and p <= r:  # square/up shapes: Triton wins; down: eager
        import monarch_triton
        out_blk, t = monarch_triton.butterfly_fwd_triton_blocked(x2, w1_bfly, w2_bfly)
        return out_blk, t
    x_r = x2.reshape(B, k, p).transpose(0, 1)
    out1 = torch.empty(B, k, q, device=x2.device, dtype=x2.dtype).transpose(0, 1)
    out1 = torch.bmm(x_r, w1_bfly.transpose(-1, -2), out=out1)
    if _USE_FAST_RIFFLE and l == k == 4 and q % 4 == 0:
        from . import monarch_riffle
        out1 = monarch_riffle.riffle_triton(out1.transpose(0, 1), B, l, q)
    else:
        out1 = (out1.transpose(0, 1).reshape(B, r, l)
                .permute(2, 0, 1).contiguous())      # riffle (eager fallback)
    out2 = torch.empty(B, l, s, device=x2.device, dtype=x2.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
    return out2.transpose(0, 1), out1                # (B, l, s) contiguous view


@torch.library.custom_op("monarch::butterfly_blk", mutates_args=())
def _butterfly_blk_op(x2: torch.Tensor, w1_bfly: torch.Tensor,
                      w2_bfly: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    out_blk, out1 = _butterfly_fwd_blocked(x2, w1_bfly, w2_bfly)
    return out_blk.contiguous(), out1


@_butterfly_blk_op.register_fake
def _(x2, w1_bfly, w2_bfly):
    B = x2.shape[0]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    return x2.new_empty(B, l, s), x2.new_empty(l, B, r)


@torch.library.custom_op("monarch::butterfly_blk_bwd", mutates_args=())
def _butterfly_blk_bwd_op(x2: torch.Tensor, w1_bfly: torch.Tensor,
                          w2_bfly: torch.Tensor, out1: torch.Tensor,
                          dout_blk: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = x2.shape[0]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    # blocked grad -> (l, B, s) strided VIEW: no gather copy needed
    dout_reshaped = dout_blk.permute(1, 0, 2)
    dw2 = torch.bmm(dout_reshaped.transpose(-1, -2), out1)
    if _USE_FAST_RIFFLE and l == k == 4 and q % 4 == 0:
        from . import monarch_riffle
        dt = torch.empty(l, B, r, device=x2.device, dtype=x2.dtype)
        torch.bmm(dout_reshaped, w2_bfly, out=dt)
        dout1 = monarch_riffle.unriffle_triton(dt, B, k, q).transpose(0, 1)
    else:
        dout1 = torch.empty(B, l, r, device=x2.device, dtype=x2.dtype).transpose(0, 1)
        dout1 = torch.bmm(dout_reshaped, w2_bfly, out=dout1)
        dout1 = (dout1.transpose(0, 1).transpose(-1, -2).contiguous()
                 .reshape(B, k, q).transpose(0, 1))  # inverse riffle (eager)
    dx = torch.empty(B, k, p, device=x2.device, dtype=x2.dtype)
    dx = (torch.bmm(dout1, w1_bfly, out=dx.transpose(0, 1))
          .transpose(0, 1).reshape(B, k * p))
    x_r = x2.reshape(B, k, p).transpose(0, 1)
    dw1 = torch.bmm(dout1.transpose(-1, -2), x_r)
    return dx, dw1, dw2


@_butterfly_blk_bwd_op.register_fake
def _(x2, w1_bfly, w2_bfly, out1, dout_blk):
    return (x2.new_empty(x2.shape), w1_bfly.new_empty(w1_bfly.shape),
            w2_bfly.new_empty(w2_bfly.shape))


def _blk_setup_context(ctx, inputs, output):
    x2, w1_bfly, w2_bfly = inputs
    ctx.save_for_backward(x2, w1_bfly, w2_bfly, output[1])


def _blk_backward(ctx, dout_blk, dout1_unused):
    x2, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
    return _butterfly_blk_bwd_op(x2, w1_bfly, w2_bfly, out1,
                                 dout_blk.contiguous())


_butterfly_blk_op.register_autograd(_blk_backward, setup_context=_blk_setup_context)


def butterfly_blk(x, w1_bfly, w2_bfly):
    """Blocked-output butterfly; interleave done here (Inductor fuses it)."""
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
        x, w1_bfly, w2_bfly = x.to(dt), w1_bfly.to(dt), w2_bfly.to(dt)
    batch_shape, n = x.shape[:-1], x.shape[-1]
    out_blk, _ = _butterfly_blk_op(x.reshape(-1, n), w1_bfly, w2_bfly)
    B, l, s = out_blk.shape
    return out_blk.transpose(1, 2).reshape(*batch_shape, s * l)


def patch_monarch_linear(use_tilelang: bool = False, blocked: bool = False,
                         use_triton: bool = False, fast_riffle: bool = False):
    """Route MonarchLinear.forward_matmul through the custom op."""
    global _USE_TILELANG, _TL, _USE_TRITON, _USE_FAST_RIFFLE
    _USE_TILELANG = use_tilelang
    _USE_TRITON = use_triton
    _USE_FAST_RIFFLE = fast_riffle
    if use_tilelang:
        import monarch_tl
        _TL = monarch_tl

    impl = butterfly_blk if blocked else butterfly

    def forward_matmul(self, x):
        output = impl(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)

    _ml.MonarchLinear.forward_matmul = forward_matmul
