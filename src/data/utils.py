from pathlib import Path
import numpy as np
from typing import Any, Dict, Union
import torch
import torch.distributed as dist
from transformers import AutoTokenizer


def get_tokenizer(args, verbose=True):
    """Get the appropriate tokenizer based on args."""
    tokenizer_name = getattr(args, 'tokenizer', 'gpt2')
    
    if tokenizer_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer_name == "mistral":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # Default tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Set model max length
    max_length = getattr(args, 'max_length', None) or getattr(args, 'sequence_length', 1024)
    tokenizer.model_max_length = max_length

    if verbose:
        print(f"Using tokenizer: {tokenizer_name}")
        print(f"Vocabulary size: {len(tokenizer)}")
        print(f"Max length: {max_length}")
        print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    return tokenizer



def get_dataset(args) -> Union[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fetch the right dataset given by the args.dataset parameter."""
    
    # Traditional datasets (your existing code)
    if args.dataset == "wikitext":
        from .wikitext import get_wikitext_data

        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        from .shakespeare import get_shakespeare_data

        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        from .arxiv import get_arxiv_2000

        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        from .arxiv import get_arxiv_full

        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        from .arxiv import get_arxiv_full
        from .wikitext import get_wikitext_data

        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        from .openwebtext2 import get_openwebtext2_data

        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        from .redpajama import get_redpajama_data

        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        from .redpajama import get_redpajamav2_data

        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        from .slimpajama import get_slimpajama_data

        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "c4":
        from .c4 import get_c4_data

        return get_c4_data(args.datasets_dir, args, 128)
    else:
        raise NotImplementedError(f"Unknown dataset key '{args.dataset}'")


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(np.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = len(self.order) // self.batch_size

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen
