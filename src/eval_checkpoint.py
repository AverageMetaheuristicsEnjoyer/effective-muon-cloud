"""Run val + LM + downstream eval on a saved checkpoint, then exit.

Loads only the model weights (not optimizer/scheduler), so the caller only
needs to match the model-shape arguments — not the optimizer choice.
"""

import os
import sys
from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
import wandb

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in [_SRC, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import distributed
from data.utils import get_tokenizer
from evals import build_evaluators
from main import define_wandb_metrics, get_args, get_data_readers, get_wandb_group
from models.utils import get_model
from optim.base import eval_and_log
from optim.utils import load_checkpoint


def main(args):
    if not args.resume_from:
        raise ValueError("--resume-from is required for eval_checkpoint.py")
    ckpt_path = Path(args.resume_from) / "main.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint file at {ckpt_path}")

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []
    args.inter_ckpts = []
    args.qargs = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    args.wandb_group = get_wandb_group(args)

    tokenizer = get_tokenizer(args)
    datareaders = get_data_readers(args, tokenizer=tokenizer)
    val_reader = datareaders["val"]
    downstream_evaluator, lm_evaluator = build_evaluators(args, tokenizer=tokenizer)

    model = get_model(args).to(args.device)
    model = distributed_backend.transform_model(model, find_unused_parameters=False)

    ckpt_iter = load_checkpoint(
        model, None, None, ckpt_path, args.device,
        load_optimizer=False, load_scheduler=False,
    )
    print(f"Loaded weights from {ckpt_path} at iter {ckpt_iter}")

    # Make eval_and_log / *.should_run pick the "final" branch (full val sweep,
    # `final-val/*` W&B metric names, downstream + lm always fire).
    args.iterations = ckpt_iter

    if distributed_backend.is_master_process() and args.wandb:
        eval_tags = list(args.wandb_tags or []) + ["eval-only"]
        wandb.init(
            project=args.wandb_project,
            name=f"{args.experiment_name}_eval@{ckpt_iter}",
            group=args.wandb_group,
            tags=eval_tags,
            config=vars(args),
        )
        define_wandb_metrics(
            downstream_evaluator=downstream_evaluator,
            lm_evaluator=lm_evaluator,
        )

    if "cuda" in args.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[args.dtype],
        )
    else:
        type_ctx = nullcontext()

    eval_and_log(
        ckpt_iter, model, val_reader, type_ctx,
        distributed_backend, args, opt=None, full_eval=True,
    )

    if downstream_evaluator is not None:
        distributed_backend.barrier()
        if distributed_backend.is_master_process():
            downstream_evaluator.evaluate(ckpt_iter, model, type_ctx, distributed_backend)
        distributed_backend.barrier()

    if lm_evaluator is not None:
        distributed_backend.barrier()
        if distributed_backend.is_master_process():
            lm_evaluator.evaluate(ckpt_iter, model, type_ctx, distributed_backend)
        distributed_backend.barrier()

    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
