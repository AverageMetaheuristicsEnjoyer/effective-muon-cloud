"""Optimized Monarch-Muon: butterfly linears + batched-Newton-Schulz Muon.

Ported from the optimization work in effective-muon-etc-main
(muon_monarch_package, rounds 1-4: 1947 -> 830.6 ms/iter on the 0.5B
llama @ 131k tokens/iter, 1xH100). The optimized configuration is:

    apply_monarch(model, nblocks=4)          # swap nn.Linear -> MonarchLinear
    patch_monarch_linear(blocked=True,       # opaque custom op, blocked layout
                         fast_riffle=True)   # Triton riffle/unriffle kernels
    MonarchMuonOptimizer(..., ns_dtype=torch.bfloat16, use_foreach=True)
    torch.compile(model)                     # butterfly is protected (opaque)

All optimized paths are numerically exact vs the reference implementation
(bitwise-equal outputs and gradients, verified per shape).
"""
from .monarch_linear import MonarchLinear, apply_monarch
from .monarch_muon import MonarchMuonOptimizer
from .monarch_ops import patch_monarch_linear

__all__ = [
    "MonarchLinear",
    "apply_monarch",
    "MonarchMuonOptimizer",
    "patch_monarch_linear",
]
