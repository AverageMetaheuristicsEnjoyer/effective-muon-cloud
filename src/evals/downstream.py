from contextlib import nullcontext

import torch
import wandb
from torch.utils.data import DataLoader

from .task_groups import resolve_downstream_tasks


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        if "cuda" in torch.device(device).type:
            return value.to(device, non_blocking=True)
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _shift_targets(input_ids, label_mask=None, attention_mask=None, ignore_index=-1):
    targets = input_ids.clone()

    if label_mask is not None:
        targets.masked_fill_(~label_mask, ignore_index)
    if attention_mask is not None:
        targets.masked_fill_(attention_mask == 0, ignore_index)

    return torch.nn.functional.pad(targets[..., 1:], (0, 1), value=ignore_index)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _select_latest_metrics(raw_metrics):
    # ai2-olmo-eval==0.8.5 emits v1/v2 metric variants for compatibility.
    # In that package, v1/v2 reflect two continuation-length normalizations around
    # leading-space handling (see olmo_eval/tasks.py ~120/~123 and metrics.py ~391),
    # and some metrics (e.g. acc/len_norm) are duplicated across both versions
    # (see metrics.py ~420). We keep only *_v2 to reduce logging noise.
    """Prefer versioned v2 metrics to reduce logging noise."""
    if not isinstance(raw_metrics, dict):
        return raw_metrics

    v2_metrics = {name: value for name, value in raw_metrics.items() if str(name).endswith("_v2")}
    if v2_metrics:
        return v2_metrics
    return raw_metrics


def _disable_metric_distributed_sync(metric):
    # Downstream eval runs on rank 0 only inside this trainer, so torchmetrics
    # must not try to sync metric state across the process group at compute().
    if hasattr(metric, "sync_on_compute"):
        metric.sync_on_compute = False
    if hasattr(metric, "_to_sync"):
        metric._to_sync = False
    return metric


class DownstreamEvaluator:
    def __init__(self, cfg, tokenizer, tokenizer_identifier):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tokenizer_identifier = tokenizer_identifier
        self.task_names = resolve_downstream_tasks(
            task_group=cfg.downstream_task_group,
            tasks=cfg.downstream_tasks,
        )
        self._task_runtimes = None

    @staticmethod
    def is_enabled(cfg):
        return bool(cfg.downstream_eval_enabled and cfg.downstream_eval_interval > 0)

    def wandb_metric_globs(self):
        return [f"downstream/{task_name}/*" for task_name in self.task_names]

    def _ensure_initialized(self):
        if self._task_runtimes is not None:
            return

        try:
            from olmo_eval import HFTokenizer, ICLMetric, build_task
        except ImportError as exc:
            raise ImportError(
                "Downstream evaluation requires the 'ai2-olmo-eval' package. "
                "Install the project requirements before enabling --downstream-eval-enabled."
            ) from exc

        eval_tokenizer = HFTokenizer(
            self.tokenizer_identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        runtimes = []
        for task_name in self.task_names:
            task_dataset = build_task(
                task_name,
                eval_tokenizer,
                model_ctx_len=self.cfg.sequence_length,
            )
            metric = ICLMetric(metric_type=task_dataset.metric_type)
            metric = _disable_metric_distributed_sync(metric)
            if hasattr(metric, "to"):
                metric = metric.to(self.cfg.device)
            dataloader = DataLoader(
                task_dataset,
                batch_size=self.cfg.eval_batch_size,
                collate_fn=task_dataset.collate_fn,
                drop_last=False,
                shuffle=False,
                num_workers=0,
            )
            runtimes.append(
                {
                    "task_name": task_name,
                    "dataset": task_dataset,
                    "metric": metric,
                    "dataloader": dataloader,
                }
            )

        self._task_runtimes = runtimes

    def should_run(self, curr_iter):
        return curr_iter == self.cfg.iterations or (
            curr_iter > 0 and curr_iter % self.cfg.downstream_eval_interval == 0
        )

    @torch.no_grad()
    def evaluate(self, curr_iter, model, type_ctx, distributed_backend):
        if not distributed_backend.is_master_process():
            return None

        self._ensure_initialized()

        was_training = model.training
        model.eval()

        logs = {
            "iter": curr_iter,
            "consumed_tokens": (
                curr_iter
                * distributed_backend.get_world_size()
                * self.cfg.acc_steps
                * self.cfg.batch_size
                * self.cfg.sequence_length
            ),
        }
        print_lines = []

        for runtime in self._task_runtimes:
            metric = runtime["metric"]
            if hasattr(metric, "reset"):
                metric.reset()

            for batch in runtime["dataloader"]:
                batch = _move_to_device(batch, self.cfg.device)
                input_ids = batch["input_ids"]
                targets = _shift_targets(
                    input_ids,
                    label_mask=batch.get("label_mask"),
                    attention_mask=batch.get("attention_mask"),
                )

                with type_ctx if type_ctx is not None else nullcontext():
                    outputs = model(input_ids, targets=targets, get_logits=True)

                metric.update(batch, outputs["logits"])

            raw_metrics = metric.compute()
            raw_metrics = _select_latest_metrics(raw_metrics)
            if not isinstance(raw_metrics, dict):
                metric_name = getattr(runtime["dataset"], "metric_type", "score")
                raw_metrics = {metric_name: raw_metrics}

            for metric_name, metric_value in raw_metrics.items():
                scalar = _to_float(metric_value)
                log_key = f"downstream/{runtime['task_name']}/{metric_name}"
                logs[log_key] = scalar
                print_lines.append(f"{log_key}={scalar:.4f}")

        print(
            f">Downstream Eval: Iter={curr_iter} "
            f"consumed_tokens={logs['consumed_tokens']} "
            + " ".join(print_lines)
        )

        if self.cfg.wandb:
            wandb.log(logs)

        if was_training:
            model.train()

        return logs
