from typing import Dict, List, Optional
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
warnings.filterwarnings('ignore')

from ..CustomGHN3.graph import Graph, GraphBatch
from .tiny_lm_fixed import TinyTransformerLM


def _valid_heads(d_model: int, n_heads: int) -> int:
    # ensure heads divide model dim
    if d_model % n_heads == 0:
        return n_heads
    for h in [16, 12, 8, 6, 4, 3, 2]:
        if h <= d_model and d_model % h == 0:
            return h
    return 2


class LMArchitectureDataset(Dataset):
    """
    Yields pairs (model, graph) for diverse Transformer LM architectures.
    Graphs are constructed with CustomGHN3.graph.Graph.
    """

    def __init__(
        self,
        num_models: int = 200,
        vocab_size: int = 50257,
        d_model_choices: Optional[List[int]] = None,
        n_layers_choices: Optional[List[int]] = None,
        n_heads_choices: Optional[List[int]] = None,
        d_ff_choices: Optional[List[int]] = None,
        max_len_choices: Optional[List[int]] = None,
        device: Optional[str] = None,
        ve_cutoff: int = 50,
        dense: bool = True,
    ):
        self.num_models = num_models
        self.vocab_size = vocab_size
        self.dense = dense
        self.ve_cutoff = ve_cutoff
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.d_model_choices = d_model_choices or [64, 128, 256, 384, 512]
        self.n_layers_choices = n_layers_choices or [2, 4, 6, 8, 12]
        self.n_heads_choices = n_heads_choices or [2, 4, 8, 16]
        self.d_ff_choices = d_ff_choices or [128, 256, 512, 1024, 2048]
        self.max_len_choices = max_len_choices or [256, 512, 1024, 2048]

        # Pre-sample configs deterministically for reproducibility
        g = torch.Generator().manual_seed(0)
        self.configs: List[Dict] = []
        for _ in range(num_models):
            d_model = int(torch.choice(torch.tensor(self.d_model_choices), generator=g))
            n_layers = int(torch.choice(torch.tensor(self.n_layers_choices), generator=g))
            n_heads = int(torch.choice(torch.tensor(self.n_heads_choices), generator=g))
            n_heads = _valid_heads(d_model, n_heads)
            d_ff = int(torch.choice(torch.tensor(self.d_ff_choices), generator=g))
            max_len = int(torch.choice(torch.tensor(self.max_len_choices), generator=g))
            self.configs.append(
                dict(
                    vocab_size=self.vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    max_len=max_len,
                    p_drop=0.1,
                )
            )

    def __len__(self) -> int:
        return self.num_models

    def __getitem__(self, idx: int):
        cfg = self.configs[idx]
        model = TinyTransformerLM(**cfg).to(self.device)
        graph = Graph(model, ve_cutoff=self.ve_cutoff, dense=self.dense, verbose=False)
        return model, graph


def build_lm_arch_dataloader(
    num_models: int = 200,
    batch_size: int = 4,
    vocab_size: int = 50257,
    device: Optional[str] = None,
    num_workers: int = 0,
    ve_cutoff: int = 50,
    dense: bool = True,
):
    """
    Returns (loader, is_dense) where each batch collate builds a GraphBatch.
    For GHN-3, dense must be True.
    """
    dataset = LMArchitectureDataset(
        num_models=num_models,
        vocab_size=vocab_size,
        device=device,
        ve_cutoff=ve_cutoff,
        dense=dense,
    )

    def collate_fn(items):
        models, graphs = zip(*items)
        gb = GraphBatch(list(graphs), dense=dense)
        return list(models), gb

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dense


__all__ = ["LMArchitectureDataset", "build_lm_arch_dataloader"]


# -------------------------------
# Curated GHN Training Set (10+ variants)
# Mini-GPT variants: layers 2-6, d_model 128-512, heads 2-8, MLP ratios
# -------------------------------

def _ensure_heads(d_model: int, heads: int) -> int:
    return heads if d_model % heads == 0 else _valid_heads(d_model, heads)


def _cfg(name: str, d_model: int, n_layers: int, n_heads: int, mlp_ratio: int, max_len: int = 512, vocab_size: int = 50257) -> Dict:
    n_heads = _ensure_heads(d_model, n_heads)
    d_ff = mlp_ratio * d_model
    return dict(
        name=name,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        p_drop=0.1,
    )


def ghn_training_variants(vocab_size: int = 50257, max_len: int = 1024) -> List[Dict]:
    """
    Returns a fixed list (10+) of curated LM configs for GHN training.
    Coverage:
      - layers: 2-6
      - d_model: 128-512
      - heads: 2-8 (validated to divide d_model)
      - mlp_ratio: 2-4 (maps to d_ff)
    """
    variants: List[Dict] = []
    variants.append(_cfg("mini-gpt-2L-128d-h2-r2", 128, 2, 2, 2, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-3L-128d-h4-r3", 128, 3, 4, 3, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-4L-128d-h8-r4", 128, 4, 8, 4, max_len, vocab_size))

    variants.append(_cfg("mini-gpt-2L-256d-h4-r2", 256, 2, 4, 2, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-3L-256d-h8-r3", 256, 3, 8, 3, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-4L-256d-h8-r4", 256, 4, 8, 4, max_len, vocab_size))

    variants.append(_cfg("mini-gpt-3L-384d-h6-r2", 384, 3, 6, 2, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-4L-384d-h8-r3", 384, 4, 8, 3, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-6L-384d-h6-r4", 384, 6, 6, 4, max_len, vocab_size))

    variants.append(_cfg("mini-gpt-4L-512d-h8-r2", 512, 4, 8, 2, max_len, vocab_size))
    variants.append(_cfg("mini-gpt-6L-512d-h8-r3", 512, 6, 8, 3, max_len, vocab_size))

    return variants


class GHNLMVariantsDataset(Dataset):
    """
    Dataset of curated LM variants for GHN training (deterministic, 10+ variants).
    Each item is (model, graph, meta) where meta contains the config including name.
    """

    def __init__(self, vocab_size: int = 50257, max_len: int = 1024, device: Optional[str] = None, ve_cutoff: int = 50, dense: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ve_cutoff = ve_cutoff
        self.dense = dense
        self.configs = ghn_training_variants(vocab_size=vocab_size, max_len=max_len)

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int):
        cfg = self.configs[idx].copy()
        name = cfg.pop("name")
        model = TinyTransformerLM(**cfg).to(self.device)
        graph = Graph(model, ve_cutoff=self.ve_cutoff, dense=self.dense, verbose=False)
        meta = {"name": name, **cfg}
        return model, graph, meta


def build_ghn_variants_dataloader(
    batch_size: int = 2,
    vocab_size: int = 50257,
    max_len: int = 1024,
    device: Optional[str] = None,
    num_workers: int = 0,
    ve_cutoff: int = 50,
    dense: bool = True,
):
    """
    DataLoader over a fixed curated set of LM variants (10+).
    Batches collate into (models, GraphBatch, metas).
    """
    dataset = GHNLMVariantsDataset(vocab_size=vocab_size, max_len=max_len, device=device, ve_cutoff=ve_cutoff, dense=dense)

    def collate_fn(items):
        models, graphs, metas = zip(*items)
        gb = GraphBatch(list(graphs), dense=dense)
        return list(models), gb, list(metas)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.configs


__all__ += ["GHNLMVariantsDataset", "build_ghn_variants_dataloader", "ghn_training_variants"]


