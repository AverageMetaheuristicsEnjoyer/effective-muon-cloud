"""Tensor-train (TT) linear layers with a fast materialized-W forward.

Wraps torchTT's LinearLayerTT (Novikov et al. 2015: the weight of a linear
layer reshaped to a tensor operator and stored as TT-matrix cores) with:

  apply_tt(model, rank_mid)     - replace nn.Linear -> TTLinear (surgery),
                                  lm_head/embeddings kept dense
  set_fused(True)  [default]    - materialized-W forward: merge the cores
                                  into dense W (~1e8 FLOPs, independent of
                                  batch), then one dense GEMM. Identical
                                  contraction network as the shipped sweep
                                  (fp32 allclose; bf16 equidistant from the
                                  fp32 ground truth), but avoids the sweep's
                                  giant batch-carrying intermediates: the
                                  naive path is ~50-150x slower per layer at
                                  LLM batch sizes and OOMs at micro-batch 16.
  set_modes(d)                  - number of TT factors per dimension (3..10);
                                  d=3 is speed-optimal, d=8 params-optimal at
                                  rank 64.

Mode factorizations must multiply to the model's feature dims:
  1280 = 8*10*16 (d=3), ...     3584 = 8*16*28 (d=3), ...
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "torchTT") not in sys.path:
    sys.path.insert(0, str(ROOT / "torchTT"))

from torchtt.nn import LinearLayerTT  # noqa: E402

MODES_BY_D = {
    3: {1280: (8, 10, 16), 3584: (8, 16, 28)},
    4: {1280: (8, 8, 5, 4), 3584: (8, 8, 8, 7)},
    5: {1280: (4, 4, 4, 4, 5), 3584: (8, 4, 4, 4, 7)},
    6: {1280: (2, 2, 4, 4, 4, 5), 3584: (2, 4, 4, 4, 4, 7)},
    7: {1280: (2, 2, 2, 2, 4, 4, 5), 3584: (2, 2, 2, 2, 4, 8, 7)},
    8: {1280: (2, 2, 2, 2, 2, 2, 4, 5), 3584: (2, 2, 2, 2, 2, 2, 8, 7)},
    9: {1280: (2, 2, 2, 2, 2, 2, 2, 2, 5), 3584: (2, 2, 2, 2, 2, 2, 2, 4, 7)},
    10: {1280: (2, 2, 2, 2, 2, 2, 2, 2, 5, 1),
         3584: (2, 2, 2, 2, 2, 2, 2, 2, 2, 7)},
}
MODES = MODES_BY_D[3]
_FUSED = True


def set_fused(flag: bool):
    global _FUSED
    _FUSED = flag


def set_modes(d: int):
    global MODES
    MODES = MODES_BY_D[d]


def _merge_cores(cores):
    """TT-matrix cores -> dense W (out_features, in_features).

    cores[i]: (R[i], n_out_i, n_in_i, R[i+1]). Cost is independent of batch.
    """
    w = cores[0]
    for c in cores[1:]:
        w = torch.tensordot(w, c, dims=([-1], [0]))
    w = w.squeeze(0).squeeze(-1)
    d = w.ndim // 2
    out_perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    w = w.permute(*out_perm).contiguous()
    o = 1
    for i in range(d):
        o *= w.shape[i]
    return w.reshape(o, -1)


def _fused_forward(self, x):
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
        cores = [c.to(dt) for c in self.cores]
        bias = self.bias.to(dt)
        x = x.to(dt)
    else:
        cores, bias = list(self.cores), self.bias
    w = _merge_cores(cores)
    batch_shape = x.shape[:-1]
    out = F.linear(x.reshape(-1, x.shape[-1]), w)
    return out.reshape(*batch_shape, -1) + bias.reshape(-1)


class TTLinear(LinearLayerTT):
    """LinearLayerTT with flat (features,) I/O + fused forward by default."""

    def __init__(self, in_features, out_features, rank_mid):
        d = len(MODES[in_features])
        rank = [1] + [rank_mid] * (d - 1) + [1]
        super().__init__(list(MODES[in_features]), list(MODES[out_features]),
                         rank, dtype=torch.float32)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        if _FUSED:
            return _fused_forward(self, x)
        batch_shape = x.shape[:-1]
        if torch.is_autocast_enabled():
            dt = torch.get_autocast_dtype("cuda")
            cores = [c.to(dt) for c in self.cores]
            bias = self.bias.to(dt)
            x = x.to(dt)
        else:
            cores, bias = list(self.cores), self.bias
        xm = x.reshape(*batch_shape, *self.size_in)
        # reference TT-matvec sweep (same math as LinearLayerTT.forward)
        result = xm.unsqueeze(-1)
        D = xm.ndim
        d = len(self.size_in)
        for c in cores:
            result = torch.tensordot(result, c, dims=([D - d, -1], [2, 0]))
        result = result.squeeze(-1) + bias
        return result.reshape(*batch_shape, -1)


def apply_tt(model: nn.Module, rank_mid: int = 64,
             exclude: tuple = ("lm_head",), verbose: bool = True):
    """Replace nn.Linear layers whose dims have mode factorizations."""
    before = sum(p.numel() for p in model.parameters())
    replacements = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(e in full for e in exclude):
                continue
            if child.in_features not in MODES or child.out_features not in MODES:
                continue
            new = TTLinear(child.in_features, child.out_features, rank_mid)
            replacements.append((parent, child_name, new))
    for parent, child_name, new in replacements:
        setattr(parent, child_name, new)
    after = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"TT applied: rank={rank_mid}, {len(replacements)} layers, "
              f"{before/1e6:.1f}M -> {after/1e6:.1f}M params "
              f"({after/before:.1%})", flush=True)
    return model
