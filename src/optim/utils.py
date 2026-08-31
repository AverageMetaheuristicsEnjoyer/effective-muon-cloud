from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import torch.distributed as dist
import math
import wandb

from tqdm.auto import trange


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def cos_inf_schedule(n_iterations, n_warmup, div_factor, final_div_factor, n_inf):
    """Cosine annealing with warmup and _constant_ final_lr after cycle ended.
    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        div_factor: initial division factor for warmup
        final_div_factor: final division factor for final lr
        n_inf: number of iterations for the final lr (constant lr after cycle ended)
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    max_lr = 1.0
    base_lr = max_lr / div_factor
    final_lr = base_lr / final_div_factor

    n_anneal_steps = n_iterations - n_inf

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / div_factor
        elif step < n_anneal_steps:
            t = (step - n_warmup) / (n_anneal_steps - n_warmup)
            lr = final_lr + 0.5 * (max_lr - final_lr) * (1 + np.cos(np.pi * t))
            return lr
        else:
            return final_lr

    return schedule


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
):
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_fract: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )

        else:
            return final_lr_factor

    return schedule


def cos_warmup_zero_schedule(n_iterations, n_warmup):
    """Linear warmup followed by cosine decay to zero."""
    if n_warmup >= n_iterations:
        raise ValueError("Warmup steps must be < total scheduler steps.")

    def schedule(step):
        if n_warmup > 0 and step < n_warmup:
            return step / n_warmup
        if step >= n_iterations:
            return 0.0

        t = (step - n_warmup) / (n_iterations - n_warmup)
        return 0.5 * (1 + np.cos(np.pi * t))

    return schedule


def inverse_sqrt_schedule(n_warmup):
    """Transformer warmup followed by inverse-square-root decay.

    The multiplier is normalised to reach 1.0 at ``n_warmup`` so ``--lr``
    remains the peak learning rate used by the rest of this training code.
    """
    if n_warmup <= 0:
        raise ValueError("inverse_sqrt requires --warmup-steps > 0")

    def schedule(step):
        step = step + 1
        if step <= n_warmup:
            return step / n_warmup
        return math.sqrt(n_warmup / step)

    return schedule


def build_scheduler(opt, args, total_steps=None):
    total_steps = args.iterations if total_steps is None else total_steps

    if args.scheduler == "none":
        return None

    if total_steps <= 0:
        raise ValueError(f"Scheduler requires a positive number of steps, got {total_steps}.")

    if args.scheduler in ["cos", "linear"]:
        assert args.warmup_steps < total_steps, "Warmup steps must be < total scheduler steps."
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=[group.get("lr", args.lr) for group in opt.param_groups],
            total_steps=total_steps,
            pct_start=args.warmup_steps / total_steps,
            anneal_strategy=args.scheduler,
            cycle_momentum=False,
            div_factor=1e2,
            final_div_factor=0.1,
        )

    if args.scheduler == "cos_zero":
        if args.warmup_steps != 0:
            raise ValueError(
                "--scheduler cos_zero does not support warmup. Set --warmup-steps 0."
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt,
            T_max=total_steps,
            eta_min=0.0,
        )

    if args.scheduler == "cos_warmup_zero":
        lambda_schedule = cos_warmup_zero_schedule(
            n_iterations=total_steps,
            n_warmup=args.warmup_steps,
        )
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)

    if args.scheduler == "inverse_sqrt":
        return torch.optim.lr_scheduler.LambdaLR(
            opt,
            inverse_sqrt_schedule(args.warmup_steps),
        )

    if args.scheduler == "cos_inf":
        assert args.warmup_steps < total_steps, "Warmup steps must be < total scheduler steps."
        lambda_schedule = cos_inf_schedule(
            n_iterations=total_steps,
            n_warmup=args.warmup_steps,
            n_inf=args.cos_inf_steps,
            div_factor=1e2,
            final_div_factor=0.1,
        )
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)

    if args.scheduler == "wsd":
        assert args.warmup_steps < total_steps, "Warmup steps must be < total scheduler steps."
        lambda_schedule = wsd_schedule(
            n_iterations=total_steps,
            n_warmup=args.warmup_steps,
            fract_decay=args.wsd_fract_decay,
            init_div_factor=1e2,
            final_lr_factor=args.wsd_final_lr_scale,
            decay_type=args.decay_type,
        )
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)

    raise NotImplementedError(f"Unknown scheduler: {args.scheduler}.")


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    cfg=None,
):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for idx in trange(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss

    return val_acc, val_loss, val_perplexity


@torch.no_grad()
def eval_sweep_dropk(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


_ARCHITECTURE_CONFIG_KEYS = (
    "model",
    "vocab_size",
    "sequence_length",
    "n_embd",
    "n_head",
    "n_layer",
    "multiple_of",
    "ffn_hidden_size",
    "attention_type",
    "tensorized_mode",
    "tensorized_rank",
    "tensorized_num_cores",
    "tensorized_causal",
    "linear_parameterization",
    "target_parameter_count",
    "target_parameter_tolerance",
    "tucker_rank",
    "tucker_ranks",
    "tucker_attention_ranks",
    "tucker_gate_up_ranks",
    "tucker_down_ranks",
    "tucker_rank_plan",
    "tucker_mode_layout",
    "tucker_progressive_stages",
    "tucker_progressive_warmup_steps",
    "tucker_progressive_seed",
    "tucker_progressive_verify_rtol",
    "tucker_equal_params",
    "tucker_forward_mode",
    "tucker_retract_every_step",
    "tucker_vector_transport",
    "tucker_riemannian_muon",
    "tucker_riemannian_muon_post_ns_project",
    "tucker_dense_adamw_matrices",
    "tucker_lr_scaling_mode",
    "tucker_lr_scaling_eps",
    "tucker_lr_scaling_power_iters",
    "tucker_lr_scaling_use_stiefel_unit_norm",
    "tucker_lr_scaling_post_ns_project",
    "tucker_lr_scaling_stiefel_drift_threshold",
    "tucker_lr_scaling_strict_bound_check",
    "tucker_lr_scaling_exact_svd_debug",
    "tucker_lr_scaling_log_interval",
)


def _architecture_metadata(model):
    config = getattr(model, "config", None)
    if config is None:
        return None
    metadata = {
        key: getattr(config, key, None) for key in _ARCHITECTURE_CONFIG_KEYS
    }
    if metadata.get("attention_type") != "tensorized":
        for key in (
            "tensorized_mode",
            "tensorized_rank",
            "tensorized_num_cores",
            "tensorized_causal",
        ):
            metadata[key] = None
    if metadata.get("linear_parameterization") != "tucker":
        for key in (
            "target_parameter_count",
            "target_parameter_tolerance",
            "tucker_rank",
            "tucker_ranks",
            "tucker_attention_ranks",
            "tucker_gate_up_ranks",
            "tucker_down_ranks",
            "tucker_rank_plan",
            "tucker_mode_layout",
            "tucker_progressive_stages",
            "tucker_progressive_warmup_steps",
            "tucker_progressive_seed",
            "tucker_progressive_verify_rtol",
            "tucker_equal_params",
            "tucker_forward_mode",
            "tucker_retract_every_step",
            "tucker_vector_transport",
            "tucker_riemannian_muon",
            "tucker_riemannian_muon_post_ns_project",
            "tucker_dense_adamw_matrices",
            "tucker_lr_scaling_mode",
            "tucker_lr_scaling_eps",
            "tucker_lr_scaling_power_iters",
            "tucker_lr_scaling_use_stiefel_unit_norm",
            "tucker_lr_scaling_post_ns_project",
            "tucker_lr_scaling_stiefel_drift_threshold",
            "tucker_lr_scaling_strict_bound_check",
            "tucker_lr_scaling_exact_svd_debug",
            "tucker_lr_scaling_log_interval",
        ):
            metadata[key] = None
    # Empty per-mode rank strings and omitted values are the same policy.
    for key in (
        "tucker_ranks",
        "tucker_attention_ranks",
        "tucker_gate_up_ranks",
        "tucker_down_ranks",
    ):
        if not metadata.get(key):
            metadata[key] = None
    tucker_stats = getattr(model, "_tucker_replacement_stats", None)
    if tucker_stats is not None:
        metadata["resolved_tucker_plans"] = tucker_stats.plans
    return metadata


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir: Path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "itr": itr,
        "architecture": _architecture_metadata(model),
        "progressive_tucker": getattr(model, "_progressive_tucker_state", None),
    }
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, ckpt_dir / "main.pt")


def load_checkpoint(
    model,
    opt,
    scheduler,
    ckpt_path,
    device,
    *,
    load_optimizer=True,
    load_scheduler=True,
):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    progressive_state = ckpt.get("progressive_tucker")
    if progressive_state is not None:
        from optim.progressive_tucker import restore_progressive_tucker_shapes_

        restore_progressive_tucker_shapes_(model, opt, progressive_state)
    saved_architecture = ckpt.get("architecture")
    current_architecture = _architecture_metadata(model)
    if saved_architecture is not None and current_architecture is not None:
        mismatches = []
        scaler_keys = {
            "tucker_lr_scaling_mode",
            "tucker_lr_scaling_eps",
            "tucker_lr_scaling_power_iters",
            "tucker_lr_scaling_use_stiefel_unit_norm",
            "tucker_lr_scaling_post_ns_project",
            "tucker_lr_scaling_stiefel_drift_threshold",
            "tucker_lr_scaling_strict_bound_check",
            "tucker_lr_scaling_exact_svd_debug",
            "tucker_lr_scaling_log_interval",
        }
        saved_scaler_mode = saved_architecture.get(
            "tucker_lr_scaling_mode",
            "none",
        )
        current_scaler_mode = current_architecture.get(
            "tucker_lr_scaling_mode",
            "none",
        )
        for key in _ARCHITECTURE_CONFIG_KEYS:
            saved_value = saved_architecture.get(key)
            current_value = current_architecture.get(key)
            if key == "tucker_riemannian_muon" and key not in saved_architecture:
                # Checkpoints written before this opt-in mode existed are
                # equivalent to an explicit false value.
                saved_value = False
            if key == "tucker_lr_scaling_mode" and key not in saved_architecture:
                saved_value = "none"
            if key == "tucker_mode_layout" and key not in saved_architecture:
                saved_value = "balanced4"
            if (
                key == "tucker_riemannian_muon_post_ns_project"
                and key not in saved_architecture
            ):
                saved_value = False
            if (
                key == "tucker_lr_scaling_post_ns_project"
                and key not in saved_architecture
            ):
                saved_value = True
            if (
                key in scaler_keys
                and saved_scaler_mode == "none"
                and current_scaler_mode == "none"
            ):
                continue
            pure_rank_only = (
                saved_architecture.get("tucker_equal_params") is False
                and current_architecture.get("tucker_equal_params") is False
            )
            if pure_rank_only and key in (
                "target_parameter_count",
                "target_parameter_tolerance",
            ):
                # These values only validate a pure model at construction;
                # they do not affect its parameters or checkpoint shapes.
                continue
            if key in (
                "tucker_ranks",
                "tucker_attention_ranks",
                "tucker_gate_up_ranks",
                "tucker_down_ranks",
            ):
                saved_value = saved_value or None
                current_value = current_value or None
            if saved_value != current_value:
                mismatches.append(
                    f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Checkpoint architecture does not match the requested model. "
                "Repeat the original tensorized/Tucker CLI flags. Mismatches: "
                + "; ".join(mismatches)
            )
    model.load_state_dict(ckpt["model"])
    if load_optimizer and opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    if load_scheduler and scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    itr = ckpt["itr"]
    return itr


def save_worker_state(ckpt_dir: Path, train_reader=None):
    # Dataloader, rng states
    worker_state = {
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_torch_gpu": torch.cuda.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    if train_reader is not None and hasattr(train_reader, "state_dict"):
        worker_state["train_reader_state"] = train_reader.state_dict()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(worker_state, ckpt_dir / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir: Path, train_reader=None):
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt", weights_only=False)
    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])

    if train_reader is not None and getattr(train_reader, "requires_checkpoint_state", False):
        if "train_reader_state" not in worker_state:
            raise RuntimeError(
                "Checkpoint is missing FineWeb train reader state. "
                "Old FineWeb streaming checkpoints are unsupported."
            )

    if train_reader is not None and "train_reader_state" in worker_state:
        if not hasattr(train_reader, "load_state_dict"):
            raise RuntimeError(
                "Checkpoint contains train reader state but the active reader "
                "does not support load_state_dict()."
            )
        train_reader.load_state_dict(worker_state["train_reader_state"])
        return True

    return False
