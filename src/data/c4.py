from tqdm import tqdm
import numpy as np
import datasets
import datasets.distributed
from transformers import AutoTokenizer
import torch.distributed as dist
import os
import glob


_hf_tknzr = None


def _get_hf_tknzr():
    global _hf_tknzr
    if _hf_tknzr is None:
        _hf_tknzr = AutoTokenizer.from_pretrained("gpt2")
    return _hf_tknzr


def _find_json_files(data_dir):
    """Find all JSON/JSON.gz files recursively under data_dir."""
    files = glob.glob(os.path.join(data_dir, "**/*.json.gz"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No JSON files found in {data_dir}")
    print(f"Found {len(files)} JSON files in {data_dir}")
    return files


def get_c4_data(datasets_dir, args, num_proc=40):
    if getattr(args, 'streaming', False):
        return get_c4_data_streaming(datasets_dir, args)
    else:
        return get_c4_data_common(datasets_dir, args, num_proc)


def get_c4_data_streaming(datasets_dir, args):
    if os.path.isdir(datasets_dir):
        json_files = _find_json_files(datasets_dir)
        data_files = {"train": json_files}

        eval_batch_size = getattr(args, 'eval_batch_size', args.batch_size)
        eval_batches = getattr(args, 'eval_batches', 32)
        val_examples_needed = int(eval_batch_size * eval_batches)

        # Shuffle before split so val gets random examples, not just the
        # first N in file order. take/skip are lazy and don't share state.
        shuffled = datasets.load_dataset(
            "json", data_files=data_files, split='train', streaming=True
        ).shuffle(seed=2357, buffer_size=10_000)

        val_dataset = shuffled.take(val_examples_needed)
        train_dataset = shuffled.skip(val_examples_needed)

        # Heuristic: ~4 bytes per token for English text in raw JSON
        estimated_tokens = sum(os.path.getsize(f) for f in json_files) // 4
    else:
        print(f"{datasets_dir} not found locally, streaming from HuggingFace...")
        train_dataset = datasets.load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        )
        val_dataset = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )
        estimated_tokens = 365_000_000_000  # C4 en train ~365B tokens

    train_dataset = train_dataset.shuffle(seed=getattr(args, 'data_seed', 1337))

    world_size, rank = 1, 0
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        train_dataset = datasets.distributed.split_dataset_by_node(
            train_dataset, rank=rank, world_size=world_size
        )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "world_size": world_size,
        "rank": rank,
        "estimated_tokens": estimated_tokens,
    }


def get_c4_data_common(datasets_dir, args, num_proc=40):
    train_bin_path = os.path.join(datasets_dir, "train.bin")
    val_bin_path = os.path.join(datasets_dir, "val.bin")

    if not os.path.exists(train_bin_path):
        os.makedirs(datasets_dir, exist_ok=True)
        json_files = _find_json_files(datasets_dir)

        dataset = datasets.load_dataset("json", data_files={"train": json_files})
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            hf_tknzr = _get_hf_tknzr()
            ids = hf_tknzr.encode(
                text=example["text"], add_special_tokens=True,
                padding=False, truncation=False,
            )
            return {"ids": ids, "len": len(ids)}

        tokenized = split_dataset.map(
            process, remove_columns=["text"],
            desc="tokenizing the splits", num_proc=num_proc,
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(datasets_dir, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": train_bin_path,
        "val": val_bin_path,
    }