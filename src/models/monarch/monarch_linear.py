"""
Monarch structured linear layer (block-diagonal butterfly decomposition).
Adapted from: https://github.com/HazyResearch/monarch

Public API
----------
MonarchLinear  – drop-in replacement for nn.Linear with butterfly decomposition
apply_monarch  – patch any model in-place: replace nn.Linear → MonarchLinear
"""
from __future__ import annotations

import gc
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(
            dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5))
        )
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output


class BlockdiagButterflyMultiply(torch.autograd.Function):
    """Fast block-diagonal butterfly matrix multiply with manual backward.

    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = int(np.prod(batch_shape))
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = int(np.prod(batch_shape))
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        dx, dw1_bfly, dw2_bfly = None, None, None
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly


blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply


class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        factory_kwargs = {'device': device, 'dtype': dtype}

        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz, **factory_kwargs))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz, **factory_kwargs))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz, **factory_kwargs))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    @property
    def saving(self):
        return (
            (self.blkdiag1.numel() + self.blkdiag2.numel())
            / (self.in_features * self.out_features)
        )

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)


# ---------------------------------------------------------------------------
# Model surgery helper
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDE = ("lm_head",)


def apply_monarch(model: nn.Module, nblocks: int = 4, exclude: tuple = _DEFAULT_EXCLUDE,
                  verbose: bool = True) -> None:
    """Replace nn.Linear layers with MonarchLinear in-place.

    Must be called **before** DDP wrapping.

    Args:
        model:   The model to modify (GPTBase, Llama, or any nn.Module).
        nblocks: Number of butterfly blocks per MonarchLinear layer.
        exclude: Substrings — any layer whose full name contains one of these
                 will be skipped (default: skip lm_head to preserve weight tying).
        verbose: Print each replacement.
    """
    before = sum(p.numel() for p in model.parameters())

    replacements = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(ex in full_name for ex in exclude):
                continue
            new_mod = MonarchLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
                nblocks=nblocks,
            )
            replacements.append((parent, child_name, child, new_mod, full_name))

    for parent, child_name, old, new, full_name in replacements:
        setattr(parent, child_name, new)
        if verbose:
            print(f"  Monarch: {full_name}  {old.weight.shape} → blkdiag {new.blkdiag1.shape}")
        del old

    gc.collect()

    after = sum(p.numel() for p in model.parameters())
    print(
        f"\nMonarch applied: nblocks={nblocks}, replaced {len(replacements)} layers\n"
        f"  Params before : {before / 1e6:.2f}M\n"
        f"  Params after  : {after / 1e6:.2f}M  ({after / before:.2%})\n"
    )
