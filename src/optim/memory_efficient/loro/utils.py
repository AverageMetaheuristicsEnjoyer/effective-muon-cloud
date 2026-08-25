import gc

import torch
import torch.nn as nn

from .lowrank_module import LowRankLinear
from .lowrank_adpt_module import LowRankAdapterLinear


def apply_loro(model, rank, init="orth", scope="all", mode="lowrank",
               alpha=1.0, init_range=0.02, verbose=True):
    """Replace nn.Linear modules with LowRank variants in-place.

    Works with any model architecture by matching module names against
    ``scope`` keywords.  Must be called **before** DDP wrapping.

    Args:
        model: Model to modify (in-place).
        rank: Low-rank dimension *r*.
        init: Weight initialisation method (see LowRankLinear / LowRankAdapterLinear).
        scope: Which sub-modules to target: ``"all"`` | ``"attn"`` | ``"mlp"``.
        mode: ``"lowrank"`` → LowRankLinear, ``"adapter"`` → LowRankAdapterLinear.
        alpha: Scaling factor (adapter mode only).
        init_range: Passed as ``init_range`` to LowRankLinear (for ``init="auto"``).
        verbose: Print each replacement.
    """
    target_keywords = {
        "all": ["attn", "mlp"],
        "attn": ["attn"],
        "mlp": ["mlp"],
    }[scope]

    full_param = sum(p.numel() for p in model.parameters())

    # Collect replacements first (avoid mutating while iterating)
    replacements = []
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child_module, nn.Linear):
                continue
            if not any(kw in full_name for kw in target_keywords):
                continue
            if mode == "adapter":
                new_module = LowRankAdapterLinear(child_module, rank, alpha, init)
            else:
                new_module = LowRankLinear(child_module, rank, init, init_range)
            replacements.append((parent_module, child_name, child_module, new_module, full_name))

    for parent, child_name, old, new, full_name in replacements:
        setattr(parent, child_name, new)
        if verbose:
            print(f"LORO: {full_name}: {old} --> {new}")
        del old
        torch.cuda.empty_cache()
        gc.collect()

    lowrank_param = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"\nLORO ({mode}) applied successfully!  rank={rank}, scope={scope}, init={init}\n"
            f"  Full-rank params : {full_param / 1e6:.2f} M\n"
            f"  Low-rank params  : {lowrank_param / 1e6:.2f} M\n"
            f"  Compression      : {lowrank_param / full_param:.4%}\n"
        )


def get_loro_param_groups(model, lr_scaler=-1, hidden_size=768, mode="lowrank"):
    """Build parameter groups expected by :class:`LOROAdamW`.

    Args:
        model: Model whose ``nn.Linear`` layers have already been replaced
            by :func:`apply_loro`.
        lr_scaler: Learning-rate scaler for low-rank params.
            ``-1`` means adaptive ``r / d`` (see LOROAdamW).
        hidden_size: Model hidden dimension (used when ``lr_scaler=-1``).
        mode: ``"lowrank"`` includes a ``"regular"`` group for non-lowrank
            params.  ``"adapter"`` freezes them instead.

    Returns:
        List of param-group dicts with a ``"type"`` key
        (``"regular"`` | ``"lowrank_in"`` | ``"lowrank_out"``).
    """
    lowrank_params_in = []
    lowrank_params_out = []

    for _name, module in model.named_modules():
        if isinstance(module, (LowRankLinear, LowRankAdapterLinear)):
            lowrank_params_in.append(module.weight_in)
            lowrank_params_out.append(module.weight_out)

    id_lowrank = {id(p) for p in lowrank_params_in + lowrank_params_out}

    param_groups = []

    if mode == "lowrank":
        # Split regular (non-lowrank) params into decay / no-decay,
        # mirroring the logic in GPTBase.get_parameter_group_specs().
        decay_params = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if id(p) in id_lowrank:
                continue
            if any(kw in name for kw in ("norm", "bias", "embed", "lm_head")):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if decay_params:
            param_groups.append({
                "type": "regular",
                "params": decay_params,
                "hidden_size": hidden_size,
            })
        if no_decay_params:
            param_groups.append({
                "type": "regular",
                "params": no_decay_params,
                "weight_decay": 0.0,
                "hidden_size": hidden_size,
            })
    else:
        # Adapter mode: freeze everything except low-rank factors
        for p in model.parameters():
            if id(p) not in id_lowrank:
                p.requires_grad_(False)

    param_groups.extend([
        {
            "type": "lowrank_in",
            "params": lowrank_params_in,
            "lr_scaler": lr_scaler,
            "hidden_size": hidden_size,
        },
        {
            "type": "lowrank_out",
            "params": lowrank_params_out,
            "lr_scaler": lr_scaler,
            "hidden_size": hidden_size,
        },
    ])

    n_regular = sum(
        sum(p.numel() for p in g["params"])
        for g in param_groups if g["type"] == "regular"
    )
    n_lowrank = sum(p.numel() for p in lowrank_params_in + lowrank_params_out)
    print(
        f"LORO param groups: {n_regular / 1e6:.2f}M regular, "
        f"{n_lowrank / 1e6:.2f}M low-rank  (lr_scaler={lr_scaler})"
    )
    return param_groups
