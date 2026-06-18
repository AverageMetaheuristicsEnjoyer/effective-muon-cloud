import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import wandb


WIKITEXT103_HF_REPO_ID = "mattdangerw/wikitext-103-raw"
WIKITEXT103_HF_ZIP_FILENAME = "wikitext-103-raw-v1.zip"


def _download_wikitext103_zip(zip_path):
    try:
        from huggingface_hub import hf_hub_download
    except Exception as error:
        raise RuntimeError(
            "huggingface_hub is required for Wikitext-103 LM eval download. "
            "Install dependencies from requirements.txt."
        ) from error

    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=WIKITEXT103_HF_REPO_ID,
                filename=WIKITEXT103_HF_ZIP_FILENAME,
                repo_type="dataset",
            )
        )
        if downloaded_path != zip_path:
            shutil.copyfile(downloaded_path, zip_path)
    except Exception as error:
        raise RuntimeError(
            "Failed to download Wikitext-103 from Hugging Face dataset "
            f"'{WIKITEXT103_HF_REPO_ID}' ({WIKITEXT103_HF_ZIP_FILENAME})."
        ) from error


def _tokenizer_cache_name(cfg, tokenizer):
    name = getattr(cfg, "tokenizer", None) or getattr(tokenizer, "name_or_path", "tokenizer")
    return str(name).replace("/", "_")


def resolve_lm_eval_datasets(datasets):
    if datasets:
        return datasets
    return ["wikitext103"]


def _prepare_wikitext103_cache(datasets_dir, cfg, tokenizer):
    base_dir = Path(datasets_dir) / "evals" / "wikitext103"
    raw_dir = base_dir / "raw"
    cache_dir = base_dir / _tokenizer_cache_name(cfg, tokenizer)
    val_bin_path = cache_dir / "val.bin"

    if val_bin_path.exists():
        return val_bin_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "wikitext-103-raw-v1.zip"
    valid_path = raw_dir / "wikitext-103-raw" / "wiki.valid.raw"

    if not valid_path.exists():
        if not zip_path.exists():
            print("Downloading Wikitext-103 raw validation data...")
            _download_wikitext103_zip(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

    with open(valid_path, "r", encoding="utf-8") as data_file:
        raw_eval_data = data_file.read()

    token_ids = tokenizer.encode(raw_eval_data, add_special_tokens=False)
    np.asarray(token_ids, dtype=np.uint16).tofile(val_bin_path)
    return val_bin_path


class SequentialTokenReader:
    def __init__(self, data_path, batch_size, sequence_length):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.step = 0
        self.data = None

        self.num_tokens = len(self._get_data())
        self.num_sequences = max(0, (self.num_tokens - 1) // self.sequence_length)
        self._num_batches = self.num_sequences // self.batch_size

    def _get_data(self):
        if self.data is None:
            self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        return self.data

    def set_step(self, step):
        self.step = step

    def num_batches(self):
        return self._num_batches

    def sample_batch(self):
        if self.step >= self._num_batches:
            raise RuntimeError("LM eval reader exhausted")

        data = self._get_data()
        start_sequence = self.step * self.batch_size
        xs = []
        ys = []

        for batch_offset in range(self.batch_size):
            sequence_index = start_sequence + batch_offset
            token_start = sequence_index * self.sequence_length
            chunk = data[token_start : token_start + self.sequence_length + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        self.step += 1
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


class AuxiliaryLMEvaluator:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset_names = resolve_lm_eval_datasets(cfg.lm_eval_datasets)
        self._readers = None

    @staticmethod
    def is_enabled(cfg):
        return bool(cfg.lm_eval_enabled and cfg.lm_eval_interval > 0)

    def wandb_metric_globs(self):
        return [f"aux-lm/{dataset_name}/*" for dataset_name in self.dataset_names]

    def _ensure_initialized(self):
        if self._readers is not None:
            return

        readers = {}
        for dataset_name in self.dataset_names:
            if dataset_name != "wikitext103":
                raise ValueError(
                    f"Unsupported LM eval dataset '{dataset_name}'. Only 'wikitext103' is implemented."
                )
            val_bin_path = _prepare_wikitext103_cache(
                self.cfg.eval_cache_dir or self.cfg.datasets_dir,
                self.cfg,
                self.tokenizer,
            )
            readers[dataset_name] = SequentialTokenReader(
                val_bin_path,
                batch_size=self.cfg.eval_batch_size,
                sequence_length=self.cfg.sequence_length,
            )

        self._readers = readers

    def should_run(self, curr_iter):
        return curr_iter == self.cfg.iterations or (
            curr_iter > 0 and curr_iter % self.cfg.lm_eval_interval == 0
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

        for dataset_name, reader in self._readers.items():
            reader.set_step(0)
            total_loss = 0.0
            num_batches = reader.num_batches()
            if num_batches == 0:
                raise ValueError(
                    f"LM eval dataset '{dataset_name}' is too small for "
                    f"batch_size={self.cfg.eval_batch_size} and sequence_length={self.cfg.sequence_length}."
                )

            for _ in range(num_batches):
                x, y = reader.sample_batch()
                if "cuda" in torch.device(self.cfg.device).type:
                    x = x.pin_memory().to(self.cfg.device, non_blocking=True)
                    y = y.pin_memory().to(self.cfg.device, non_blocking=True)
                else:
                    x = x.to(self.cfg.device)
                    y = y.to(self.cfg.device)

                with type_ctx:
                    outputs = model(x, targets=y, get_logits=False)
                total_loss += float(outputs["loss"].detach().cpu().item())

            mean_loss = total_loss / num_batches
            perplexity = math.exp(mean_loss)
            logs[f"aux-lm/{dataset_name}/loss"] = mean_loss
            logs[f"aux-lm/{dataset_name}/perplexity"] = perplexity
            print_lines.append(
                f"aux-lm/{dataset_name}/loss={mean_loss:.4f} "
                f"aux-lm/{dataset_name}/perplexity={perplexity:.4f}"
            )

        print(
            f">Aux LM Eval: Iter={curr_iter} "
            f"consumed_tokens={logs['consumed_tokens']} "
            + " ".join(print_lines)
        )

        if self.cfg.wandb:
            wandb.log(logs)

        if was_training:
            model.train()

        return logs
