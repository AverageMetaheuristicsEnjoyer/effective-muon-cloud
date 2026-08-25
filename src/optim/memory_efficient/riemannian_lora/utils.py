import gc
import torch
import torch.nn as nn

from .riemannian_linear import RiemannianLoraLinear
from .riemannian_optimizer import RiemannianPretrainOptimizer


_SCOPE_KEYWORDS = {
    "all":  ["attn", "mlp"],
    "attn": ["attn"],
    "mlp":  ["mlp"],
}


def apply_riemannian_lora(model, rank: int, scope: str = "all", init: str = "orth", verbose: bool = True):
    """Replace nn.Linear modules with RiemannianLoraLinear in-place.

    Must be called **before** DDP wrapping.

    Args:
        model:  The model to modify.
        rank:   Low-rank dimension *r* (stored as 2r internally).
        scope:  Which sub-modules to target: ``"all"`` | ``"attn"`` | ``"mlp"``.
        init:   Weight init style for B factor: ``"orth"`` (Kaiming-scaled)
                or ``"zero"`` (zero init, adapter-style).
        verbose: Print each replacement.
    """
    keywords = _SCOPE_KEYWORDS[scope]
    full_params = sum(p.numel() for p in model.parameters())

    replacements = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if not any(kw in full_name for kw in keywords):
                continue
            new_mod = RiemannianLoraLinear(child, rank=rank, init=init)
            replacements.append((parent, child_name, child, new_mod, full_name))

    for parent, child_name, old, new, full_name in replacements:
        setattr(parent, child_name, new)
        if verbose:
            print(f"RiemannianLora: {full_name}: {old} → {new}")
        del old
        torch.cuda.empty_cache()
        gc.collect()

    lr_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nRiemannianLora applied: rank={rank}, scope={scope}, init={init}\n"
        f"  Full-rank params : {full_params / 1e6:.2f} M\n"
        f"  Low-rank params  : {lr_params / 1e6:.2f} M\n"
        f"  Compression      : {lr_params / full_params:.4%}\n"
    )


def get_riemannian_param_groups(model, lr: float, betas, weight_decay: float):
    """Build (lora_groups, regular_groups) for RiemannianPretrainOptimizer.

    Each LoRA group covers one RiemannianLoraLinear module and carries the
    ``param_names`` key required by RiemannianLora.step().

    Regular groups (embeddings, norms, biases, lm_head) go to AdamW.

    Returns:
        lora_groups     – list of param-group dicts for RiemannianLora
        regular_groups  – list of param-group dicts for AdamW
    """
    lora_factor_ids: set = set()
    lora_groups = []

    for _name, module in model.named_modules():
        if isinstance(module, RiemannianLoraLinear):
            a_w = module.lora_A.weight
            b_w = module.lora_B.weight
            lora_factor_ids.add(id(a_w))
            lora_factor_ids.add(id(b_w))
            lora_groups.append({
                "params": [a_w, b_w],
                "param_names": ["lora_A.weight", "lora_B.weight"],
                "lr": lr,
                "betas": list(betas),
            })

    decay_params, no_decay_params = [], []
    _NO_DECAY_KW = ("norm", "bias", "embed", "lm_head")

    for pname, p in model.named_parameters():
        if id(p) in lora_factor_ids:
            continue
        if any(kw in pname for kw in _NO_DECAY_KW):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    regular_groups = []
    if decay_params:
        regular_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        regular_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    n_lora = sum(p.numel() for g in lora_groups for p in g["params"])
    n_reg  = sum(p.numel() for g in regular_groups for p in g["params"])
    print(
        f"Riemannian param groups: {n_lora / 1e6:.2f}M LoRA-factor, "
        f"{n_reg / 1e6:.2f}M regular"
    )
    return lora_groups, regular_groups


def build_riemannian_optimizer(model, args, opt_type: str = "adamw") -> RiemannianPretrainOptimizer:
    """Convenience builder called from main.py."""
    rank = args.riemannian_rank if args.riemannian_rank > 0 else int(args.density * args.n_embd)

    apply_riemannian_lora(
        model,
        rank=rank,
        scope=args.riemannian_scope,
        init=args.riemannian_init,
    )

    lora_groups, regular_groups = get_riemannian_param_groups(
        model,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    return RiemannianPretrainOptimizer(
        lora_groups=lora_groups,
        regular_groups=regular_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
        opt_type=opt_type,
    )
