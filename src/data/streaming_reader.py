import itertools
import torch
from torch.utils.data import IterableDataset
from typing import Dict, Iterator


class PackedStreamingDataset(IterableDataset):
    """
    Streaming dataset that tokenizes on-the-fly and packs tokens into
    fixed-length blocks — matching the contiguous-stream approach of the
    traditional DataReader.  No padding, no truncation, zero token waste.
    """

    def __init__(self, dataset, tokenizer, seq_len, tokenize_batch_size=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.block_size = seq_len + 1  # +1 for the target shift
        self.tokenize_batch_size = tokenize_batch_size
        self.eos_token_id = tokenizer.eos_token_id

    def _flush_texts(self, texts, buffer):
        """Batch-tokenize texts and extend buffer with tokens + EOS per doc."""
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for token_ids in encoded:
            if token_ids:
                buffer.extend(token_ids)
                buffer.append(self.eos_token_id)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            iter_data = itertools.islice(
                self.dataset, worker_info.id, None, worker_info.num_workers
            )
        else:
            iter_data = iter(self.dataset)

        buffer = []
        text_batch = []

        for example in iter_data:
            text = example.get("text", "")
            if not text or not text.strip():
                continue
            text_batch.append(text)

            if len(text_batch) >= self.tokenize_batch_size:
                self._flush_texts(text_batch, buffer)
                text_batch = []

                while len(buffer) >= self.block_size:
                    chunk = buffer[: self.block_size]
                    buffer = buffer[self.block_size :]
                    t = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": t[:-1], "labels": t[1:]}

        # Flush remaining texts
        if text_batch:
            self._flush_texts(text_batch, buffer)

        # Yield remaining complete blocks, discard tail
        while len(buffer) >= self.block_size:
            chunk = buffer[: self.block_size]
            buffer = buffer[self.block_size :]
            t = torch.tensor(chunk, dtype=torch.long)
            yield {"input_ids": t[:-1], "labels": t[1:]}


class StreamingDataReader:
    """
    Streaming data reader with token packing.
    Compatible with the existing DataReader interface (sample_batch, set_step, etc.).
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size: int,
        seq_len: int,
        seed: int = 1337,
        num_workers: int = 8,
        is_eval: bool = False,
        eval_batches: int = 32,
        estimated_tokens: int = 100_000_000,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.is_eval = is_eval
        self.eval_batches = eval_batches

        self.iterable_dataset = PackedStreamingDataset(
            dataset, tokenizer, seq_len
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.iterable_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        self.step = 0

        if is_eval:
            self.num_tokens = eval_batches * batch_size * seq_len
            self._num_batches = eval_batches
        else:
            self.num_tokens = estimated_tokens
            self._num_batches = estimated_tokens // (batch_size * seq_len)

        self._dataloader_iter = None

    def _get_dataloader_iter(self):
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.dataloader)
        return self._dataloader_iter

    def sample_batch(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                batch = next(self._get_dataloader_iter())
                self.step += 1
                return batch["input_ids"], batch["labels"]
            except StopIteration:
                self._dataloader_iter = None
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Streaming dataset exhausted after "
                        f"{max_retries} restarts"
                    )

    def set_step(self, step):
        self.step = step

    def num_batches(self):
        return self._num_batches

    def __len__(self):
        return max(1, self.num_tokens - self.seq_len - 1)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Streaming datasets don't support indexing. Use sample_batch()."
        )
