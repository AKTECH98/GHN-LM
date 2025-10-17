import os
import warnings
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress warnings
warnings.filterwarnings('ignore')


def _collate_wikitext2(features):
    """Collate function for WikiText-2 batches."""
    columns = ["input_ids", "attention_mask", "labels"]
    batch = {}
    for k in columns:
        batch[k] = torch.stack([f[k] for f in features])
    return batch


def _tokenize_function(examples, tokenizer, add_eos: bool):
    # Tokenize a batch of texts
    output = tokenizer(examples["text"])  # no truncation here; we will chunk after concat
    if add_eos and tokenizer.eos_token_id is not None:
        # Append EOS to each sequence to help separation when concatenating
        output["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in output["input_ids"]]
        output["attention_mask"] = [mask + [1] for mask in output["attention_mask"]]
    return output


def _group_texts(examples, block_size: int):
    # Concatenate then split into fixed-size blocks
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // block_size) * block_size
    
    # Create result with input_ids and attention_mask (but not labels yet)
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items() if k != "labels"  # Exclude labels from initial creation
    }
    # Labels are next-token prediction targets (shift by 1 position)
    labels = []
    for input_ids in result["input_ids"]:
        # Shift labels by 1 position: labels[i] = input_ids[i+1]
        # Last position gets -100 (ignore index)
        label_seq = input_ids[1:] + [-100]
        labels.append(label_seq)
    result["labels"] = labels
    return result


def build_wikitext2(
    tokenizer_name: str = "gpt2",
    seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 2,
    cache_dir: Optional[str] = None,
) -> Dict:
    """
    Build WikiText-2 train/validation DataLoaders with fixed-length blocks.

    Returns dict with: tokenizer, vocab_size, train_loader, val_loader.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)

    add_eos = True

    tokenized = raw.map(
        lambda batch: _tokenize_function(batch, tokenizer, add_eos),
        batched=True,
        num_proc=None,
        remove_columns=["text"],
    )

    # Group into contiguous blocks of seq_len
    lm_ds = tokenized.map(
        lambda batch: _group_texts(batch, seq_len),
        batched=True,
        num_proc=None,
    )

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "labels"]
    lm_ds["train"].set_format(type="torch", columns=columns)
    lm_ds["validation"].set_format(type="torch", columns=columns)

    train_loader = DataLoader(
        lm_ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_wikitext2,
    )

    val_loader = DataLoader(
        lm_ds["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_wikitext2,
    )

    return {
        "tokenizer": tokenizer,
        "vocab_size": len(tokenizer),
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


__all__ = ["build_wikitext2"]


