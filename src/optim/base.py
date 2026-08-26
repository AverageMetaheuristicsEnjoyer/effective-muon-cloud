from contextlib import nullcontext
import copy
import json
import shutil
from pathlib import Path
import time
import yaml
import os

import torch
import wandb
from tqdm import tqdm

from dtype_utils.dtypes import (
    print_gradient_dtypes,
    print_activation_dtypes,
    print_memory_usage,
    print_optimizer_dtypes,
)

from logger.logger import DynamicsLogger
from .spectral_tracking import (
    is_spectral_snapshot_step,
    log_spectrum_and_stable_rank,
)
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    build_scheduler,
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)


def _sanitize_remote_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def _sanitize_wandb_artifact_name(value):
    return _sanitize_remote_name(value)


def _get_inter_ckpt_artifact_name(curr_iter):
    if wandb.run is None:
        raise RuntimeError("W&B run is not initialized for intermediate checkpoint upload.")

    run_group = wandb.run.group or "ungrouped"
    return _sanitize_wandb_artifact_name(
        f"{run_group}-{wandb.run.name}-inter-ckpt-{curr_iter}"
    )


def _get_inter_ckpt_hf_path(curr_iter: int, cfg):
    run_group = getattr(cfg, "wandb_group", None) or "ungrouped"
    return "/".join(
        (
            "intermediate-checkpoints",
            _sanitize_remote_name(run_group),
            _sanitize_remote_name(cfg.experiment_name),
            f"inter-ckpt-{curr_iter}",
        )
    )


def _upload_inter_ckpt_to_wandb(ckpt_dir: Path, curr_iter: int, cfg):
    if wandb.run is None:
        raise RuntimeError("W&B run is not initialized for intermediate checkpoint upload.")

    artifact_name = _get_inter_ckpt_artifact_name(curr_iter)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        description="Intermediate training checkpoint.",
        metadata={
            "iteration": curr_iter,
            "experiment_name": cfg.experiment_name,
            "wandb_group": wandb.run.group,
            "wandb_run_id": wandb.run.id,
        },
    )
    artifact.add_dir(str(ckpt_dir))
    wandb.run.log_artifact(artifact)
    artifact.wait()
    return artifact_name


def _upload_inter_ckpt_to_huggingface(ckpt_dir: Path, curr_iter: int, cfg):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for intermediate checkpoint upload to Hugging Face."
        ) from exc

    api = HfApi(endpoint="https://huggingface.co")
    path_in_repo = _get_inter_ckpt_hf_path(curr_iter, cfg)
    api.create_repo(repo_id=cfg.hf_inter_ckpt_repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=cfg.hf_inter_ckpt_repo_id,
        repo_type="model",
        folder_path=str(ckpt_dir),
        path_in_repo=path_in_repo,
        commit_message=(
            f"Upload intermediate checkpoint for {cfg.experiment_name} at iteration {curr_iter}"
        ),
    )
    return f"{cfg.hf_inter_ckpt_repo_id}/{path_in_repo}"


def _upload_inter_ckpt_and_maybe_delete(ckpt_dir: Path, curr_iter: int, cfg):
    uploaded_locations = {}
    failed_uploads = {}

    for destination in cfg.upload_inter_ckpts_to:
        try:
            if destination == "wandb":
                uploaded_locations[destination] = _upload_inter_ckpt_to_wandb(
                    ckpt_dir, curr_iter, cfg
                )
            elif destination == "huggingface":
                uploaded_locations[destination] = _upload_inter_ckpt_to_huggingface(
                    ckpt_dir, curr_iter, cfg
                )
            else:  # pragma: no cover
                raise ValueError(
                    f"Unsupported intermediate checkpoint upload destination: {destination}"
                )
        except Exception as exc:
            failed_uploads[destination] = exc

    for destination, location in uploaded_locations.items():
        remote_name = "W&B artifact" if destination == "wandb" else "Hugging Face path"
        print(
            f"Uploaded intermediate checkpoint at iter {curr_iter} "
            f"to {remote_name} '{location}'."
        )

    deleted_local_copy = False

    if uploaded_locations and cfg.delete_local_inter_ckpts_after_upload:
        shutil.rmtree(ckpt_dir)
        deleted_local_copy = True
        print(
            "Deleted local intermediate checkpoint after at least one successful "
            f"upload: {ckpt_dir}"
        )

    for destination, exc in failed_uploads.items():
        remote_name = "W&B artifact" if destination == "wandb" else "Hugging Face"
        local_state = (
            "Local checkpoint was deleted because another upload destination succeeded."
            if deleted_local_copy
            else f"Keeping local checkpoint at {ckpt_dir}."
        )
        print(
            f"WARNING: failed to upload intermediate checkpoint at iter {curr_iter} "
            f"to {remote_name}. {local_state} Error: {exc}"
        )

    return uploaded_locations, failed_uploads


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
    downstream_evaluator=None,
    lm_evaluator=None,
    activation_dtypes=None,
    debug_dtypes=False
):
    not_compiled_model = model
    progressive_controller = None
    if getattr(cfg, "tucker_progressive_stages", None):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            raise ValueError(
                "Progressive Tucker rank growth changes parameter shapes and "
                "therefore requires the single-process backend (omit "
                "--distributed-backend and launch with python, not torchrun)."
            )
        if cfg.compile:
            raise ValueError("Progressive Tucker rank growth is incompatible with --compile")
        if cfg.opt != "tensorion":
            raise ValueError("Progressive Tucker rank growth currently requires --opt tensorion")
        if getattr(cfg, "linear_parameterization", None) != "tucker":
            raise ValueError(
                "--tucker-progressive-stages requires --linear-parameterization tucker"
            )
        from optim.progressive_tucker import ProgressiveTuckerController

        raw_progressive_model = distributed_backend.get_raw_model(model)
        progressive_controller = ProgressiveTuckerController(
            raw_progressive_model,
            opt,
            cfg.tucker_progressive_stages,
            warmup_steps=cfg.tucker_progressive_warmup_steps,
            seed=cfg.tucker_progressive_seed,
            verify_rtol=cfg.tucker_progressive_verify_rtol,
        )
        print("\nProgressive Tucker schedule:")
        for line in progressive_controller.summary_lines():
            print(f"  {line}")
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    train_reader, val_reader = datareaders["train"], datareaders["val"]
    loaded_train_reader_state = False

    if cfg.resume_from:
        # By default this is a full resume including the model weights,
        # optimizer, scheduler, dataloader state, random seed, etc.
        # --decay-from-checkpoint keeps the same continuation point but
        # rebuilds the scheduler from the checkpoint LR.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
            load_scheduler=not cfg.decay_from_checkpoint,
        )
        loaded_train_reader_state = load_worker_state(
            ckpt_dir,
            train_reader=train_reader,
        )
        if cfg.decay_from_checkpoint:
            remaining_steps = cfg.iterations - curr_iter
            if remaining_steps <= 0:
                raise ValueError(
                    f"--decay-from-checkpoint requires target --iterations ({cfg.iterations}) "
                    f"to be greater than checkpoint iteration ({curr_iter})."
                )
            scheduler = build_scheduler(opt, cfg, total_steps=remaining_steps)
    else:
        curr_iter = 0

    if progressive_controller is not None:
        progressive_controller.resume_(curr_iter)

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    raw_model = distributed_backend.get_raw_model(not_compiled_model)
    flops_per_token = raw_model.num_fwd_flops + raw_model.num_bck_flops

    substep = curr_iter * cfg.acc_steps
    if not loaded_train_reader_state:
        train_reader.set_step(substep)
    stats = {
        "train_loss": [],
        "val_loss": [],
        "val_pp": [],
        "val_acc": [],
        "downstream": [],
        "aux_lm": [],
    }
    best_val_loss = float("inf")
    best_val_iter = None
    best_val_ckpt_dir = None
    best_val_metadata_path = None
    save_best_val_checkpoint = getattr(cfg, "save_best_val_checkpoint", False)
    if save_best_val_checkpoint and exp_dir is not None:
        best_val_ckpt_dir = exp_dir / "ckpts" / "best_val"
        best_val_metadata_path = best_val_ckpt_dir / "metrics.json"
        if best_val_metadata_path.exists():
            with best_val_metadata_path.open("r", encoding="utf-8") as handle:
                best_val_metadata = json.load(handle)
            best_val_loss = float(best_val_metadata["val_loss"])
            best_val_iter = int(best_val_metadata["itr"])
            print(
                "Loaded best validation checkpoint metadata: "
                f"val_loss={best_val_loss:.6f} at iter={best_val_iter}"
            )
    inter_ckpt_steps = set(cfg.inter_ckpts)
    model.train()

    # Initialize the progress bar
    if distributed_backend.is_master_process():
        pbar = tqdm(total=cfg.iterations, desc="Training Progress", position=curr_iter)
    else:
        pbar = None

    if cfg.torch_profiling and distributed_backend.is_master_process():
        from torch.profiler import ProfilerActivity, schedule

        profiler_dir = exp_dir / "profiler"

        def _on_trace_ready(p):
            profiler_dir.mkdir(exist_ok=True)
            print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32))
            trace_path = profiler_dir / f"step{p.step_num}.chrome_trace.json.gz"
            p.export_chrome_trace(str(trace_path))
            print(f"[profiler] Chrome trace saved to {trace_path}")

        _torch_profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=5, active=3, repeat=1),
            on_trace_ready=_on_trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )
    else:
        import contextlib
        _torch_profiler = contextlib.nullcontext()

    _prof = _torch_profiler.__enter__()

    # MEMORY_BENCH: lists that accumulate one entry per measurement point
    if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
        _mem_before_batch    = []
        _mem_before_step_fwd = []
        _mem_after_step_fwd  = []
        _mem_after_step_bwd  = []
        _mem_after_batch     = []
        _mem_peak_step       = []

    _time_bench = os.environ.get("TIME_BENCH", "false") in ["1", "True", "true"]
    if _time_bench:
        _t_data = []
        _t_fwd  = []
        _t_bwd  = []
        _t_opt  = []

    while curr_iter <= cfg.iterations:
        if progressive_controller is not None:
            growth = progressive_controller.maybe_grow(curr_iter)
            if growth is not None:
                # Parameter-based FLOP estimates are cached by the model.
                flops_per_token = raw_model.num_fwd_flops + raw_model.num_bck_flops
                print(
                    "Progressive Tucker growth: "
                    f"stage={growth['stage_index']} iter={curr_iter} "
                    f"parameters={growth['parameters']:,} "
                    f"target={growth['target_parameters']:,} "
                    f"relative_function_error="
                    f"{growth['max_relative_function_error']:.3e}"
                )
                if cfg.wandb and distributed_backend.is_master_process():
                    wandb.log(
                        {
                            "iter": curr_iter,
                            "tucker/progressive_stage": growth["stage_index"],
                            "tucker/progressive_parameters": growth["parameters"],
                            "tucker/progressive_target_parameters": growth[
                                "target_parameters"
                            ],
                            "tucker/progressive_function_error": growth[
                                "max_relative_function_error"
                            ],
                        }
                    )
        # Save permanent checkpoint
        if curr_iter > 0 and cfg.permanent_ckpt_interval > 0 and exp_dir is not None:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir, train_reader=train_reader)

        # Save explicit intermediate checkpoints
        if curr_iter > 0 and curr_iter in inter_ckpt_steps and exp_dir is not None:
            ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
            if distributed_backend.is_master_process():
                save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
            save_worker_state(ckpt_dir, train_reader=train_reader)

            if cfg.upload_inter_ckpts_to:
                distributed_backend.barrier()
                if distributed_backend.is_master_process():
                    _upload_inter_ckpt_and_maybe_delete(ckpt_dir, curr_iter, cfg)
                distributed_backend.barrier()

        # Save temporary checkpoint for resuming training
        if curr_iter > 0 and cfg.latest_ckpt_interval > 0 and exp_dir is not None:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir, train_reader=train_reader)

        ws = distributed_backend.get_world_size()
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_result = eval_and_log(
                curr_iter,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )
            if eval_result is not None:
                val_loss, val_pp, val_acc = eval_result
                stats["val_loss"].append(val_loss)
                stats["val_pp"].append(val_pp)
                stats["val_acc"].append(val_acc)

                if (
                    save_best_val_checkpoint
                    and best_val_ckpt_dir is not None
                    and val_loss < best_val_loss
                ):
                    previous_best = best_val_loss
                    best_val_loss = float(val_loss)
                    best_val_iter = curr_iter
                    print(
                        "Saving best validation checkpoint: "
                        f"val_loss={best_val_loss:.6f} at iter={best_val_iter} "
                        f"(previous={previous_best:.6f})"
                    )
                    save_checkpoint(
                        model,
                        opt,
                        scheduler,
                        curr_iter,
                        best_val_ckpt_dir,
                    )
                    save_worker_state(
                        best_val_ckpt_dir,
                        train_reader=train_reader,
                    )
                    with best_val_metadata_path.open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(
                            {
                                "itr": best_val_iter,
                                "val_loss": best_val_loss,
                                "val_perplexity": float(val_pp),
                                "val_accuracy": float(val_acc),
                            },
                            handle,
                            indent=2,
                            sort_keys=True,
                        )
                        handle.write("\n")
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": curr_iter,
                                "checkpoint/best_val_loss": best_val_loss,
                                "checkpoint/best_val_iter": best_val_iter,
                            }
                        )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        stable_rank_due = is_spectral_snapshot_step(
            curr_iter,
            cfg.stable_rank_interval,
        )
        spectrum_interval = (
            cfg.spectrum_interval
            if cfg.spectrum_interval > 0
            else cfg.stable_rank_interval
        )
        spectrum_due = (
            cfg.stable_rank_log_spectrum
            and is_spectral_snapshot_step(curr_iter, spectrum_interval)
        )
        if (
            cfg.wandb
            and distributed_backend.is_master_process()
            and (stable_rank_due or spectrum_due)
        ):
            log_spectrum_and_stable_rank(
                raw_model,
                curr_iter,
                log_stable_rank=stable_rank_due,
                log_full_spectrum=spectrum_due,
            )

        if downstream_evaluator is not None and downstream_evaluator.should_run(curr_iter):
            downstream_logs = None
            distributed_backend.barrier()
            if distributed_backend.is_master_process():
                downstream_logs = downstream_evaluator.evaluate(
                    curr_iter,
                    model,
                    type_ctx,
                    distributed_backend,
                )
            distributed_backend.barrier()
            if downstream_logs is not None:
                stats["downstream"].append(downstream_logs)

        if lm_evaluator is not None and lm_evaluator.should_run(curr_iter):
            lm_logs = None
            distributed_backend.barrier()
            if distributed_backend.is_master_process():
                lm_logs = lm_evaluator.evaluate(
                    curr_iter,
                    model,
                    type_ctx,
                    distributed_backend,
                )
            distributed_backend.barrier()
            if lm_logs is not None:
                stats["aux_lm"].append(lm_logs)

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # MEMORY_BENCH: Before batch
        # Content: weights + opt states
        if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
            if distributed_backend.is_master_process():
                memory_usage = torch.cuda.memory_allocated() // 1024 ** 2
                _mem_before_batch.append(memory_usage)

        # Train model
        t_start = time.perf_counter_ns()
        if "cuda" in cfg.device:
            torch.cuda.reset_peak_memory_stats()

        # FP8 weight-cache management: recompute FP8 weight scales on the first
        # microbatch of every optimizer step; reuse cached scales for the rest.
        _use_fp8 = getattr(cfg, "fp8", False) or getattr(cfg, "fp8_optim", False)
        if _use_fp8:
            import sys as _sys, os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from third_party.coat.utils._fp8manager import FP8Manager
            FP8Manager.is_first_microbatch = True

        # MEMORY_BENCH: Before batch (alternative)
        # I'm just curious whether these is different from the above 
        if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
            if distributed_backend.is_master_process():
                memory_usage = torch.cuda.memory_allocated() // 1024 ** 2
                assert _mem_before_batch[-1] == memory_usage, "THEY ARE DIFFERENT"

        if _time_bench:
            _iter_t_events = []

        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            if _time_bench:
                _te0 = torch.cuda.Event(enable_timing=True)
                _te1 = torch.cuda.Event(enable_timing=True)
                _te2 = torch.cuda.Event(enable_timing=True)
                _te3 = torch.cuda.Event(enable_timing=True)
                _te0.record()
            x, y = get_batch(train_reader, device=cfg.device)
            if _time_bench:
                _te1.record()
            # MEMORY_BENCH: Before step forward
            # Content: weights + opt states + grads (from prev microbatches)
            if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
                if distributed_backend.is_master_process():
                    memory_usage = torch.cuda.memory_allocated() // 1024 ** 2
                    _mem_before_step_fwd.append(memory_usage)
                    # Sampled allocations miss short-lived fused-kernel
                    # workspaces. Reset here so max_memory_allocated captures
                    # the true forward+backward peak of this microstep.
                    torch.cuda.reset_peak_memory_stats()
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            if _time_bench:
                _te2.record()

            # MEMORY_BENCH: After step forward
            # Content: weights + opt states + grads (from prev microbatches) + activations
            if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
                if distributed_backend.is_master_process():
                    memory_usage = torch.cuda.memory_allocated() // 1024 ** 2
                    _mem_after_step_fwd.append(memory_usage)

            loss = outputs["loss"] / cfg.acc_steps
            with type_ctx:
                loss.backward()

            if _time_bench:
                _te3.record()
                _iter_t_events.append((_te0, _te1, _te2, _te3))

            substep += 1

            # MEMORY_BENCH: After step backward
            # Content: weights + opt states + grads
            if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
                if distributed_backend.is_master_process():
                    memory_usage = torch.cuda.memory_allocated() // 1024 ** 2
                    _mem_after_step_bwd.append(memory_usage)
                    _mem_peak_step.append(
                        torch.cuda.max_memory_allocated() // 1024 ** 2
                    )

            # After first microbatch: subsequent microsteps reuse cached FP8 scales
            if _use_fp8 and microstep_idx == 0:
                FP8Manager.is_first_microbatch = False

        if cfg.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        if cfg.opt == "SFAdamW":
            opt.train()
        
        # MEMORY_BENCH: After batch
        # Content: weights + opt states + accumulated grads
        if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
            if distributed_backend.is_master_process():
                memory_usage = torch.cuda.memory_allocated() // 1024 ** 2
                _mem_after_batch.append(memory_usage)
        
        if _time_bench:
            _te4 = torch.cuda.Event(enable_timing=True)
            _te5 = torch.cuda.Event(enable_timing=True)
            _te4.record()
        opt.step()
        tucker_retraction_diagnostics = None
        if getattr(cfg, "tucker_retract_every_step", False):
            from models.tucker_linear import retract_tucker_modules_

            next_iter = curr_iter + 1
            should_compute_diagnostics = bool(
                cfg.log_interval and next_iter % cfg.log_interval == 0
            )
            tucker_retraction_diagnostics = retract_tucker_modules_(
                distributed_backend.get_raw_model(not_compiled_model),
                optimizer=opt,
                transport_optimizer_state=getattr(
                    cfg, "tucker_vector_transport", False
                ),
                compute_diagnostics=should_compute_diagnostics,
            )
        if debug_dtypes and curr_iter == 1:
            print_gradient_dtypes(model, distributed_backend)
            print_optimizer_dtypes(opt, distributed_backend)
            print_activation_dtypes(activation_dtypes, distributed_backend)
            print_memory_usage(distributed_backend, cfg.device, label="After Step 1")
        if _time_bench:
            _te5.record()
            torch.cuda.synchronize()
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)
        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        if _time_bench:
            _t_data.append(sum(e0.elapsed_time(e1) for e0, e1, _, _ in _iter_t_events))
            _t_fwd.append(sum(e1.elapsed_time(e2) for _, e1, e2, _ in _iter_t_events))
            _t_bwd.append(sum(e2.elapsed_time(e3) for _, _, e2, e3 in _iter_t_events))
            _t_opt.append(_te4.elapsed_time(_te5))

        curr_iter += 1
        if distributed_backend.is_master_process():
            pbar.update(1)

        if os.environ.get("MEMORY_BENCH", "false") in ["1", "True", "true"]:
            if curr_iter == 10 and distributed_backend.is_master_process():
                import math
                n_fwd = len(_mem_after_step_fwd)  # = 10 * acc_steps
                total_memory      = _mem_after_step_fwd[-1]
                peak_step_memory  = max(_mem_peak_step)
                model_memory      = _mem_before_batch[0]
                activation_memory = _mem_after_step_fwd[0] - _mem_before_step_fwd[0]
                optimizer_memory  = _mem_after_batch[1] - _mem_after_batch[0]
                gradient_memory   = _mem_after_step_fwd[-1] - _mem_after_step_fwd[-n_fwd // 10]
                assert math.isclose(
                    total_memory,
                    model_memory + activation_memory + optimizer_memory + gradient_memory,
                    rel_tol=1e-2, abs_tol=100,
                ), (
                    f"Memory breakdown mismatch: total={total_memory:.1f} MB "
                    f"!= sum={model_memory + activation_memory + optimizer_memory + gradient_memory:.1f} MB"
                )
                print(
                    f"\n[MEMORY BENCH] Statistics collected over 10 steps\n"
                    f"  Model Memory     : {model_memory:8.1f} MB  ({100*model_memory/total_memory:5.1f}%)\n"
                    f"  Activation Memory: {activation_memory:8.1f} MB  ({100*activation_memory/total_memory:5.1f}%)\n"
                    f"  Optimizer Memory : {optimizer_memory:8.1f} MB  ({100*optimizer_memory/total_memory:5.1f}%)\n"
                    f"  Gradient Memory  : {gradient_memory:8.1f} MB  ({100*gradient_memory/total_memory:5.1f}%)\n"
                    f"  -----------------------------------\n"
                    f"  Total            : {total_memory:8.1f} MB\n"
                    f"  Peak microstep   : {peak_step_memory:8.1f} MB\n"
                )
                exit(0)

        if _time_bench:
            if curr_iter == 10 and distributed_backend.is_master_process():
                n = len(_t_data) - 1  # 9 iterations (skip iter 0)
                a_data = sum(_t_data[1:]) / n
                a_fwd  = sum(_t_fwd[1:]) / n
                a_bwd  = sum(_t_bwd[1:]) / n
                a_opt  = sum(_t_opt[1:]) / n
                a_total = a_data + a_fwd + a_bwd + a_opt
                print(
                    f"\n[TIME BENCH] Average per iteration (iters 1-9, iter 0 warmup skipped)\n"
                    f"  Data Loading  : {a_data:8.2f} ms  ({100*a_data/a_total:5.1f}%)\n"
                    f"  Forward       : {a_fwd:8.2f} ms  ({100*a_fwd/a_total:5.1f}%)\n"
                    f"  Backward      : {a_bwd:8.2f} ms  ({100*a_bwd/a_total:5.1f}%)\n"
                    f"  Optimizer Step: {a_opt:8.2f} ms  ({100*a_opt/a_total:5.1f}%)\n"
                    f"  -----------------------------------\n"
                    f"  Total         : {a_total:8.2f} ms\n"
                    f"\n  Note: Total excludes grad clipping, zero_grad, scheduler,\n"
                    f"  WA/EMA, and torch.cuda.synchronize() overhead.\n"
                )
                exit(0)

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            stats["train_loss"].append(train_loss)
            consumed_tokens = curr_iter * ws * cfg.acc_steps * cfg.batch_size * cfg.sequence_length

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if "cuda" in cfg.device else 0.0
            reserved_mem_gb = torch.cuda.memory_reserved() / 1e9 if "cuda" in cfg.device else 0.0

            print(
                f"Train: Iter={curr_iter} "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e} "
                f"peak_mem={peak_mem_gb:.2f}GB "
                f"consumed_tokens={consumed_tokens}"
            )

            if cfg.wandb:
                logs = {
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    "iter_dt": dt,
                    "consumed_tokens": consumed_tokens,
                    "throughput/total_training_gflops": flops_per_token * consumed_tokens / 1e9,
                    "tok_gpu_sec": cfg.sequence_length * cfg.batch_size * cfg.acc_steps / dt,
                    "grad_norm": grad_norm,
                    "memory/peak_allocated_gb": peak_mem_gb,
                    "memory/reserved_gb": reserved_mem_gb,
                }
                if tucker_retraction_diagnostics is not None:
                    logs.update(
                        {
                            "tucker/retracted_modules": (
                                tucker_retraction_diagnostics["modules"]
                            ),
                            "tucker/retracted_factors": (
                                tucker_retraction_diagnostics["factors"]
                            ),
                            "tucker/transported_core_momenta": (
                                tucker_retraction_diagnostics["transported_cores"]
                            ),
                            "tucker/transported_factor_momenta": (
                                tucker_retraction_diagnostics["transported_factors"]
                            ),
                            "tucker/retraction_max_orthogonality_error": (
                                tucker_retraction_diagnostics.get(
                                    "max_orthogonality_error", 0.0
                                )
                            ),
                            "tucker/retraction_mean_orthogonality_error": (
                                tucker_retraction_diagnostics.get(
                                    "mean_orthogonality_error", 0.0
                                )
                            ),
                            "tucker/transport_max_momentum_tangency_error": (
                                tucker_retraction_diagnostics.get(
                                    "max_momentum_tangency_error", 0.0
                                )
                            ),
                            "tucker/transport_mean_momentum_tangency_error": (
                                tucker_retraction_diagnostics.get(
                                    "mean_momentum_tangency_error", 0.0
                                )
                            ),
                        }
                    )
                tucker_lr_scaling_metrics = getattr(
                    opt,
                    "last_tucker_lr_scaling_metrics",
                    None,
                )
                if tucker_lr_scaling_metrics:
                    logs.update(tucker_lr_scaling_metrics)
                wandb.log(logs)

        if _prof is not None:
            _prof.step()

    _torch_profiler.__exit__(None, None, None)
    return stats


def eval_and_log(
    curr_iter,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return None

    model.eval()
    if cfg.opt == "SFAdamW":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} "
        f"consumed_tokens={curr_iter * distributed_backend.get_world_size() * cfg.acc_steps * cfg.batch_size * cfg.sequence_length} "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                "consumed_tokens": curr_iter * distributed_backend.get_world_size() * cfg.acc_steps * cfg.batch_size * cfg.sequence_length,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                "consumed_tokens": curr_iter * distributed_backend.get_world_size() * cfg.acc_steps * cfg.batch_size * cfg.sequence_length,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
    return val_loss, val_perplexity, val_acc
