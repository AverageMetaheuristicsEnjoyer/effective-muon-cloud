import argparse
import json
import sys
import os
from pathlib import Path
import random

import numpy as np
import torch
import wandb

# Add src/ to path so all modules resolve correctly when running from project root
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in [_SRC, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from data.utils import DataReader, get_dataset, get_tokenizer
from data.fineweb import build_fineweb_readers
import distributed
from evals import build_evaluators
from models.utils import get_model
from optim.base import train
from optim.utils import build_scheduler
from optim.optimization import get_optimizer

from data.streaming_reader import StreamingDataReader

from dtype_utils.dtypes import (
    register_activation_hooks,
    print_model_dtypes,
    print_memory_usage,
)


def validate_checkpoint_args(args):
    upload_destinations = [] if args.upload_inter_ckpts_to is None else list(
        dict.fromkeys(args.upload_inter_ckpts_to)
    )
    args.upload_inter_ckpts_to = upload_destinations

    if args.delete_local_inter_ckpts_after_upload and not upload_destinations:
        raise ValueError(
            "--delete-local-inter-ckpts-after-upload requires "
            "--upload-inter-ckpts-to."
        )

    if "wandb" in upload_destinations and not args.wandb:
        raise ValueError("--upload-inter-ckpts-to wandb requires --wandb.")

    if "huggingface" in upload_destinations:
        args.hf_inter_ckpt_repo_id = (
            args.hf_inter_ckpt_repo_id or os.environ.get("HF_INTER_CKPT_REPO_ID")
        )
        if not args.hf_inter_ckpt_repo_id:
            raise ValueError(
                "--upload-inter-ckpts-to huggingface requires "
                "--hf-inter-ckpt-repo-id or HF_INTER_CKPT_REPO_ID."
            )

    inter_ckpts = [] if args.inter_ckpts is None else args.inter_ckpts
    if not inter_ckpts:
        if upload_destinations:
            raise ValueError("--upload-inter-ckpts-to requires --inter-ckpts.")
        if args.delete_local_inter_ckpts_after_upload:
            raise ValueError(
                "--delete-local-inter-ckpts-after-upload requires --inter-ckpts."
            )
        args.inter_ckpts = []
        return

    if args.permanent_ckpt_interval > 0:
        raise ValueError(
            "--inter-ckpts is mutually exclusive with --permanent-ckpt-interval."
        )

    if args.no_local_save:
        raise ValueError("--inter-ckpts requires local checkpoint saving.")

    last_step = None
    seen_steps = set()
    for step in inter_ckpts:
        if step <= 0:
            raise ValueError("--inter-ckpts must contain only positive iterations.")
        if step > args.iterations:
            raise ValueError(
                f"--inter-ckpts step {step} exceeds --iterations ({args.iterations})."
            )
        if step in seen_steps:
            raise ValueError("--inter-ckpts must be unique.")
        if last_step is not None and step <= last_step:
            raise ValueError("--inter-ckpts must be sorted in strictly increasing order.")
        seen_steps.add(step)
        last_step = step

    args.inter_ckpts = inter_ckpts


def define_wandb_metrics(downstream_evaluator=None, lm_evaluator=None):
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")
    wandb.define_metric("final-val/*", step_metric="iter")
    wandb.define_metric("memory/*", step_metric="iter")
    wandb.define_metric("throughput/*", step_metric="iter")
    wandb.define_metric("stable_rank/*", step_metric="iter")
    wandb.define_metric("stable_rank_norm/*", step_metric="iter")
    wandb.define_metric("lr", step_metric="iter")
    wandb.define_metric("iter_dt", step_metric="iter")
    wandb.define_metric("tok_gpu_sec", step_metric="iter")
    wandb.define_metric("grad_norm", step_metric="iter")
    wandb.define_metric("consumed_tokens", step_metric="iter")

    if downstream_evaluator is not None:
        for metric_glob in downstream_evaluator.wandb_metric_globs():
            wandb.define_metric(metric_glob, step_metric="iter")

    if lm_evaluator is not None:
        for metric_glob in lm_evaluator.wandb_metric_globs():
            wandb.define_metric(metric_glob, step_metric="iter")


def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []
    validate_checkpoint_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    # ── FP8 setup ────────────────────────────────────────────────────────────
    args.qargs = None
    if args.fp8 or args.fp8_optim:
        import sys as _sys
        _sys.path.insert(0, _ROOT)  # ensure third_party is findable
        from third_party.coat.utils._fp8_quantization_config import QuantizationConfig
        args.qargs = QuantizationConfig(
            quantize_model="coat_real" if args.fp8 else "none",
            fabit=args.fp8_fabit,
            fwbit=args.fp8_fwbit,
            fobit=args.fp8_fobit,
            babit=args.fp8_babit,
            bwbit=args.fp8_bwbit,
            bobit=args.fp8_bobit,
            group_size=args.fp8_group_size,
            weight_memory_efficient=args.fp8_weight_memory_efficient,
            first_order_expansion=args.fp8_expansion,
            second_order_expansion=args.fp8_expansion,
            first_order_bit=args.fp8_first_order_bit,
            second_order_bit=args.fp8_second_order_bit,
            qgroup_size=args.fp8_qgroup_size,
        )
        if distributed_backend.is_master_process():
            print(f"\nFP8 QuantizationConfig:\n{args.qargs}\n")

    # ── Experiment naming / WandB ─────────────────────────────────────────
    exp_name = get_exp_name(args)
    wandb_group = get_wandb_group(args)
    args.wandb_group = wandb_group
    exp_dir = Path(args.results_base_folder) / wandb_group / exp_name if not args.no_local_save else None
    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    runtime_tokenizer = None
    needs_runtime_tokenizer = getattr(args, "streaming", False) or (
        getattr(args, "downstream_eval_enabled", False)
        and getattr(args, "downstream_eval_interval", 0) > 0
    ) or (
        getattr(args, "lm_eval_enabled", False)
        and getattr(args, "lm_eval_interval", 0) > 0
    )
    if needs_runtime_tokenizer:
        runtime_tokenizer = get_tokenizer(args)

    datareaders = get_data_readers(args, tokenizer=runtime_tokenizer)
    downstream_evaluator, lm_evaluator = build_evaluators(
        args,
        tokenizer=runtime_tokenizer,
    )

    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            group=wandb_group,
            tags=args.wandb_tags,
            config=vars(args),
        )
        define_wandb_metrics(
            downstream_evaluator=downstream_evaluator,
            lm_evaluator=lm_evaluator,
        )

    model = get_model(args).to(args.device)
    print(f"\nModel:\n{model}")

    if args.opt == "monarch_muon" or args.apply_monarch:
        from models.monarch import apply_monarch, patch_monarch_linear

        apply_monarch(model, nblocks=args.monarch_nblocks)
        patch_monarch_linear(blocked=True, fast_riffle=True)

    # ── Tensor-train linears: replace nn.Linear with TTLinear *before* DDP ──
    if args.use_tt:
        from models.tt_integration import apply_tt, set_fused, set_modes
        set_modes(args.tt_modes_d)
        set_fused(True)
        apply_tt(model, rank_mid=args.tt_rank)
        model = model.to(args.device)

    # ── Riemannian LoRA: replace Linear modules with RiemannianLoraLinear *before* DDP ──
    if args.opt in ("riemannian_adamw", "riemannian_sgd"):
        from optim.memory_efficient.riemannian_lora import apply_riemannian_lora
        riemannian_rank = args.riemannian_rank if args.riemannian_rank > 0 else int(args.density * args.n_embd)
        apply_riemannian_lora(
            model,
            rank=riemannian_rank,
            scope=args.riemannian_scope,
            init=args.riemannian_init,
        )

    # ── LORO: replace Linear modules with LowRank *before* DDP ────────
    if args.opt in ("loro", "loro_adpt"):
        from optim.memory_efficient.loro.utils import apply_loro
        loro_mode = "adapter" if args.opt == "loro_adpt" else "lowrank"
        loro_rank = args.loro_rank if args.loro_rank > 0 else int(args.density * args.n_embd)
        apply_loro(
            model,
            rank=loro_rank,
            init=args.loro_init,
            scope=args.loro_scope,
            mode=loro_mode,
            alpha=args.loro_alpha,
            init_range=args.init_std,
        )
        
    # ── DEBUG ─────────────────────────────────────────────
    if args.debug_dtype:
        print_model_dtypes(model, distributed_backend)
        activation_dtypes = register_activation_hooks(model, distributed_backend)

    else:
        activation_dtypes = None


    model = distributed_backend.transform_model(
        model,
        find_unused_parameters=(args.opt == "badam"),
    )

    # ── Parameter groups ───────────────────────────────────────────────
    if args.opt == "monarch_muon":
        group_specs = None
        optimized_params_cnt = sum(
            p.numel()
            for p in distributed_backend.get_raw_model(model).parameters()
            if p.requires_grad
        )
    elif args.opt in ("riemannian_adamw", "riemannian_sgd"):
        # Param groups are built inside the optimizer instantiation block below.
        group_specs = None
        optimized_params_cnt = sum(p.numel() for p in distributed_backend.get_raw_model(model).parameters() if p.requires_grad)

    elif args.opt in ("loro", "loro_adpt"):
        # LORO uses its own param-group structure (type: regular/lowrank_in/lowrank_out)
        from optim.memory_efficient.loro.utils import get_loro_param_groups
        loro_mode = "adapter" if args.opt == "loro_adpt" else "lowrank"
        group_specs = get_loro_param_groups(
            distributed_backend.get_raw_model(model),
            lr_scaler=args.loro_lr_scaler,
            hidden_size=args.n_embd,
            mode=loro_mode,
        )
        optimized_params_cnt = sum(
            p.numel() for g in group_specs for p in g["params"]
        )
    else:
        group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
        param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
        optimized_params_cnt = 0
        for g in group_specs:
            params = []
            for p_name in g["params"]:
                translated_p_names = (
                    distributed_backend.translate_model_parameter_name_for_node(p_name)
                )
                params += [param_name_mapping[p_name] for p_name in translated_p_names]
            g["params"] = params
            optimized_params_cnt += sum([p.numel() for p in g["params"]])

    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log({"parameters": params_cnt, "optimized_parameters": optimized_params_cnt})

    # ── Optimiser ─────────────────────────────────────────────────────────
    if args.opt in ("riemannian_adamw", "riemannian_sgd"):
        from optim.memory_efficient.riemannian_lora import get_riemannian_param_groups, RiemannianPretrainOptimizer
        riemannian_rank = args.riemannian_rank if args.riemannian_rank > 0 else int(args.density * args.n_embd)
        lora_groups, regular_groups = get_riemannian_param_groups(
            distributed_backend.get_raw_model(model),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        opt_type = "sgd" if args.opt == "riemannian_sgd" else "adamw"
        opt = RiemannianPretrainOptimizer(
            lora_groups=lora_groups,
            regular_groups=regular_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
            opt_type=opt_type,
        )
    elif args.opt == "coat_adamw":
        from third_party.coat.optimizer.fp8_adamw import CoatAdamW
        if args.qargs is None:
            raise ValueError("coat_adamw requires --fp8-optim (which builds qargs).")
        opt = CoatAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            qargs=args.qargs,
        )
    elif args.opt == "triton_coat_adamw":
        from third_party.coat.optimizer.triton_fp8_adamw import TritonCoatAdamW
        if args.qargs is None:
            raise ValueError("triton_coat_adamw requires --fp8-optim (which builds qargs).")
        opt = TritonCoatAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            qargs=args.qargs,
        )
    elif args.opt == "solo_adamw":
        from third_party.solo.adamw import AdamWQ
        opt = AdamWQ(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            bits=tuple(args.solo_bits),
            quantile=args.solo_quantile,
            block_sizes=tuple(args.solo_block_sizes),
            quantizers=tuple(args.solo_quantizers),
        )
    elif args.opt == "solo_triton_adamw":
        from third_party.solo.triton.adamw import TritonSoloAdamW
        opt = TritonSoloAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            quantile=args.solo_quantile,
            block_size=args.solo_block_sizes[0],
        )
    elif args.opt in ("muon", "muonlite", "numuon"):
        from third_party.lite.muonlite import MuonLite, NuMuon
        raw_model = distributed_backend.get_raw_model(model)
        muon_params = []
        adamw_params = []
        for name, p in raw_model.named_parameters():
            translated = distributed_backend.translate_model_parameter_name_for_node(name)
            for t_name in translated:
                t_param = param_name_mapping[t_name]
                if t_param.ndim == 2 and not any(k in name for k in ("wte", "lm_head", "embed")):
                    muon_params.append((name, t_param))
                else:
                    adamw_params.append((name, t_param))
        if args.opt == "muonlite":
            optimizer_cls = MuonLite
            lite_kwargs = dict(
                beta1=args.lite_beta1, beta2=args.lite_beta2,
                chi=args.lite_chi, chi_adamw=args.lite_chi_adamw,
                subspace_ratio=args.lite_subspace_ratio,
            )
        elif args.opt == "numuon":
            optimizer_cls = NuMuon
            lite_kwargs = dict(
                numuon_rank_start=args.numuon_rank_start,
                numuon_rank_end=args.numuon_rank_end,
                numuon_rank_scheduler=args.numuon_rank_scheduler,
                numuon_rank_warmup_fraction=args.numuon_rank_warmup_fraction,
                numuon_rank_decay_end_fraction=args.numuon_rank_decay_end_fraction,
                numuon_krylov_iters=args.numuon_krylov_iters,
                numuon_oversample=args.numuon_oversample,
                numuon_warm_start=args.numuon_warm_start,
                numuon_fast=args.numuon_fast,
                numuon_lmo_log_interval=args.numuon_lmo_log_interval,
                numuon_lmo_output_path=(
                    str(exp_dir / "numuon_lmo.jsonl")
                    if exp_dir is not None and args.numuon_lmo_log_interval > 0
                    else None
                ),
                numuon_lmo_max_per_group=args.numuon_lmo_max_per_group,
                numuon_lmo_groups=args.numuon_lmo_groups,
            )
        else:
            optimizer_cls = MuonLite
            # Vanilla Muon: disable LITE (no subspace, no amplification, no damping)
            lite_kwargs = dict(
                beta1=0.0, beta2=0.0,
                chi=1.0, chi_adamw=1.0,
                subspace_ratio=0.0,
            )
        opt = optimizer_cls(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ns_steps=args.lite_ns_steps,
            muon_theta=args.lite_muon_theta,
            muon_num_splits=args.muon_num_splits,
            muon_split_dim=args.muon_split_dim,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            total_steps=args.iterations,
            warmup_steps=args.warmup_steps,
            **lite_kwargs,
        )
    elif args.opt == "monarch_muon":
        from models.monarch import MonarchMuonOptimizer

        raw_model = distributed_backend.get_raw_model(model)
        muon_params = []
        adamw_params = []
        for name, parameter in raw_model.named_parameters():
            if parameter.ndim >= 2 and not any(
                excluded in name for excluded in ("wte", "wpe", "lm_head", "embed")
            ):
                muon_params.append(parameter)
            else:
                adamw_params.append(parameter)
        opt = MonarchMuonOptimizer(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=args.lr,
            momentum=args.momentum,
            adamw_betas=(args.beta1, args.beta2),
            adamw_weight_decay=args.weight_decay,
            adamw_eps=args.eps,
            ns_dtype=torch.bfloat16,
            use_foreach=True,
        )
    elif args.opt in ("loro", "loro_adpt"):
        from optim.memory_efficient.loro import LOROAdamW
        opt = LOROAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            correct_bias=True,
            no_deprecation_warning=True,
            loro_type=args.loro_type,
            model=distributed_backend.get_raw_model(model),
            use_exact_loro=args.use_exact_loro,
        )
    elif args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "SFAdamW":
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )
    elif args.opt == "sgd":
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = get_optimizer(group_specs, args, model=model, qargs=args.qargs)  # Passes qargs, currently not implemented

    print(f"\nOptimizer:\n{opt}")

    # ── DEBUG ────────────────────────────────────
    if args.debug_dtype:
        print_memory_usage(distributed_backend, args.device, label="After Init")

    # ── LR scheduler ──────────────────────────────────────────────────────
    scheduler = build_scheduler(opt, args)

    # ── Auto-resume ───────────────────────────────────────────────────────
    if exp_dir is not None and (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"Experiment dir {exp_dir} already exists. "
                "Set --auto-resume to resume, or use a different --experiment-name."
            )
        else:
            args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif distributed_backend.is_master_process() and exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)

    if args.decay_from_checkpoint and args.resume_from is None:
        raise ValueError(
            "--decay-from-checkpoint requires --resume-from or an auto-resume checkpoint."
        )
    if args.decay_from_checkpoint and args.warmup_steps != 0:
        raise ValueError(
            "--decay-from-checkpoint requires --warmup-steps 0 so the rebuilt scheduler "
            "starts from the checkpoint LR."
        )

    # ── Train ─────────────────────────────────────────────────────────────
    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
        downstream_evaluator=downstream_evaluator,
        lm_evaluator=lm_evaluator,
        activation_dtypes=activation_dtypes,
        debug_dtypes=args.debug_dtype,
    )

    # We don't need to save such complex structure as it's fields are already in args
    del args.qargs
    stats["args"] = vars(args)
    if distributed_backend.is_master_process() and exp_dir is not None:
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args):
    """Returns the experiment run name (from --experiment-name)."""
    return args.experiment_name


def get_wandb_group(args):
    """Returns the wandb group: explicit --wandb-group or auto-generated from model/data/iterations."""
    if args.wandb_group is not None:
        return args.wandb_group
    return f"{args.model}-{args.n_layer}L{args.n_head}H_{args.dataset}_{args.iterations // 1000}k"


def get_data_readers(args, verbose=True, tokenizer=None):
    """Get data readers, supporting both streaming and traditional approaches."""

    # Set eval_batch_size if not provided
    if not hasattr(args, 'eval_batch_size') or args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Set workers if not provided
    if not hasattr(args, 'workers'):
        args.workers = 8

    if args.dataset in ("fineweb", "fineweb-edu"):
        tokenizer = tokenizer or get_tokenizer(args)
        tokenizer_factory = lambda: get_tokenizer(args, verbose=False)
        return build_fineweb_readers(
            args,
            tokenizer=tokenizer,
            tokenizer_factory=tokenizer_factory,
            verbose=verbose,
        )

    data_srcs = get_dataset(args)

    # Check if we're using streaming datasets
    if isinstance(data_srcs, dict) and "train_dataset" in data_srcs:
        # Streaming approach
        print("Setting up streaming data readers...")

        tokenizer = tokenizer or get_tokenizer(args)
        estimated_tokens = data_srcs.get("estimated_tokens", 100_000_000)

        train_reader = StreamingDataReader(
            dataset=data_srcs["train_dataset"],
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            seq_len=args.sequence_length,
            seed=args.data_seed,
            num_workers=args.workers,
            is_eval=False,
            estimated_tokens=estimated_tokens,
        )

        val_reader = StreamingDataReader(
            dataset=data_srcs["val_dataset"],
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            seq_len=args.sequence_length,
            seed=args.data_seed,
            num_workers=0,
            is_eval=True,
            eval_batches=args.eval_batches,
        )

        if verbose:
            print("Using streaming data readers")
            print(f"Train reader: batch_size={args.batch_size}, seq_len={args.sequence_length}")
            print(f"Val reader: batch_size={args.eval_batch_size}, seq_len={args.sequence_length}")
            print(f"Eval batches: {args.eval_batches}")

        return {"train": train_reader, "val": val_reader}
    
    else:
        # Traditional approach (your existing code)
        train_reader = DataReader(
            data_src=data_srcs["train"],
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seed=args.data_seed,
            with_replacement=False,
            auto_shard=True,
            keep_in_ram=args.data_in_ram,
        )
        val_reader = DataReader(
            data_src=data_srcs["val"],
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seed=args.data_seed,
            with_replacement=False,
            auto_shard=False,
            keep_in_ram=args.data_in_ram,
        )
        
        if verbose:
            print(f"Num training tokens: {train_reader.num_tokens}")
            print(f"Num validation tokens: {val_reader.num_tokens}")
        
        return {"train": train_reader, "val": val_reader}



if __name__ == "__main__":
    args = get_args()
    main(args)
