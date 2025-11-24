"""
Language Model Architecture Loader for GHN-3 Training.

This module provides comprehensive language model architecture datasets for training
Graph HyperNetworks (GHN-3) on language modeling tasks. It supports:

- Custom GPT-based Transformer variants (GPT Encoder, Mini GPT)
- Comprehensive parameter space exploration
"""

from typing import Dict, List, Optional
import warnings
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
warnings.filterwarnings('ignore')

from GHN.graph import Graph, GraphBatch

# Import the available model architectures
from LM import (
    GPTEncoderLayerLM, GPTEncoderConfig,
    GPTDecoderLM, MiniGPTConfig
)


# =============================================================================
# Utility Functions
# =============================================================================

def _valid_heads(d_model: int, n_heads: int) -> int:
    """Ensure attention heads divide model dimension."""
    if d_model % n_heads == 0:
        return n_heads
    for h in [16, 12, 8, 6, 4, 3, 2]:
        if h <= d_model and d_model % h == 0:
            return h
    return 2


def _get_valid_heads(d_model: int, max_heads: int = 32) -> List[int]:
    """Get all valid head counts that divide d_model."""
    valid_heads = []
    for n_heads in range(1, min(max_heads + 1, d_model + 1)):
        if d_model % n_heads == 0:
            valid_heads.append(n_heads)
    return valid_heads


def create_model_from_config(model_type: str, config: Dict, device: str = 'cpu'):
    """
    Factory function to create different types of language models.
    
    Args:
        model_type: Type of model ('gpt_encoder', 'mini_gpt')
        config: Model configuration dictionary
        device: Device to create model on
    
    Returns:
        Instantiated model on specified device
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    config_copy = config.copy()
    
    # Remove metadata fields that shouldn't be passed to model constructor
    metadata_fields = ['name', 'model_type']
    for field in metadata_fields:
        config_copy.pop(field, None)
    
    # Parameter mapping for different model types
    if model_type in ['gpt_encoder', 'mini_gpt']:
        # Transformer-based models: map parameters
        _map_transformer_params(config_copy)
        model = _create_transformer_model(model_type, config_copy)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: gpt_encoder, mini_gpt")
    
    return model.to(device)


def _map_transformer_params(config: Dict) -> None:
    """Map parameters for Transformer-based models."""
    if 'n_layers' in config:
        config['n_layer'] = config.pop('n_layers')
    if 'n_heads' in config:
        config['n_head'] = config.pop('n_heads')
    if 'max_len' in config:
        config['max_seq_len'] = config.pop('max_len')
    if 'mlp_ratio' in config:
        config['d_ff'] = int(config['mlp_ratio'] * config['d_model'])
        config.pop('mlp_ratio')


def _create_transformer_model(model_type: str, config: Dict):
    """Create Transformer-based model."""
    if model_type == 'gpt_encoder':
        return GPTEncoderLayerLM(GPTEncoderConfig(**config))
    elif model_type == 'mini_gpt':
        return GPTDecoderLM(MiniGPTConfig(**config))
    else:
        raise ValueError(f"Unknown Transformer model type: {model_type}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "GHNLMVariantsDataset", 
    "build_ghn_variants_dataloader", 
    "ghn_training_variants",
    "create_model_from_config"
]


# =============================================================================
# Configuration Generation Functions
# =============================================================================

def _ensure_heads(d_model: int, heads: int) -> int:
    """Ensure attention heads divide model dimension."""
    return heads if d_model % heads == 0 else _valid_heads(d_model, heads)


def _create_base_config(name: str, model_type: str, d_model: int, n_layers: int, 
                       n_heads: int, mlp_ratio: float, max_len: int = 512, 
                       vocab_size: int = 50257) -> Dict:
    """Create base configuration dictionary."""
    n_heads = _ensure_heads(d_model, n_heads)
    d_ff = int(mlp_ratio * d_model)  # Ensure d_ff is an integer
    
    config = {
        'name': name,
        'model_type': model_type,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'd_ff': d_ff,
        'max_len': max_len,
        'p_drop': 0.1,
    }
    
    # Set tie_weights to False for all model types for GHN compatibility
    config['tie_weights'] = False  # Important for GHN compatibility
    
    return config


# =============================================================================
# Model Configuration Generation
# =============================================================================

def ghn_training_variants(vocab_size: int = 50257, max_len: int = 1024, 
                          max_d_model: Optional[int] = None,
                          max_layers: Optional[int] = None) -> List[Dict]:
    """
    Generate comprehensive language model configurations for GHN training.
    
    Creates ALL possible combinations of model architectures with the following coverage:
    - Model types: GPT Encoder, Mini GPT
    - Layers: 1-20 (extensive range)
    - Dimensions: 32-2048 (comprehensive range of model sizes)
    - Attention heads: 1-32 (all valid divisors of d_model)
    - MLP ratios: 1.0-8.5 (all reasonable feed-forward ratios)
    
    Args:
        vocab_size: Vocabulary size for language models
        max_len: Maximum sequence length
        max_d_model: Maximum d_model for GPT Encoder/Mini GPT variants (filters out larger models).
                    Default None = no limit. Suggested: 512 for <40GB memory, 768 for <80GB
        max_layers: Maximum layers for GPT Encoder/Mini GPT variants (filters out deeper models).
                   Default None = no limit. Suggested: 8 for <40GB memory, 12 for <80GB
        
    Returns:
        List of model configuration dictionaries
    """
    variants: List[Dict] = []
    
    # Define parameter ranges
    d_models = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    layer_counts = list(range(1, 21))  # 1-20 layers
    mlp_ratios = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]
    
    # Filter parameter ranges based on max_d_model and max_layers
    if max_d_model is not None:
        d_models = [d for d in d_models if d <= max_d_model]
    if max_layers is not None:
        layer_counts = [l for l in layer_counts if l <= max_layers]
    
    # Generate Transformer variants
    variants.extend(_generate_transformer_variants(d_models, layer_counts, mlp_ratios, max_len, vocab_size))
    
    return variants


def _generate_transformer_variants(d_models: List[int], layer_counts: List[int], 
                                  mlp_ratios: List[float], max_len: int, vocab_size: int) -> List[Dict]:
    """Generate Transformer model variants."""
    variants = []
    
    for d_model in d_models:
        for n_layers in layer_counts:
            valid_heads = _get_valid_heads(d_model, max_heads=32)
            
            for n_heads in valid_heads:
                for mlp_ratio in mlp_ratios:
                    # Generate all variants without size limits
                    name = f"gpt-encoder-{n_layers}L-{d_model}d-h{n_heads}-r{mlp_ratio}"
                    variants.append(_create_base_config(name, "gpt_encoder", d_model, n_layers, n_heads, mlp_ratio, max_len, vocab_size))
                    
                    name = f"mini-gpt-{n_layers}L-{d_model}d-h{n_heads}-r{mlp_ratio}"
                    variants.append(_create_base_config(name, "mini_gpt", d_model, n_layers, n_heads, mlp_ratio, max_len, vocab_size))
    
    return variants


# =============================================================================
# Dataset Classes
# =============================================================================

class GHNLMVariantsDataset(Dataset):
    """
    Comprehensive dataset of language model variants for GHN training.
    
    This dataset provides access to all possible combinations of language model
    architectures including GPT-based Transformer variants.
    Each item returns (model, graph, meta) where:
    - model: Instantiated language model
    - graph: Graph representation for GHN-3
    - meta: Metadata including configuration and model type
    
    IMPORTANT: This dataset uses LAZY LOADING - models are created on-demand
    in __getitem__() when requested, not pre-loaded in __init__(). This ensures
    memory efficiency when dealing with large datasets (e.g., 100K+ models).
    Only model configurations (dictionaries) are stored in memory.
    """

    def __init__(self, vocab_size: int = 50257, max_len: int = 1024, 
                 device: Optional[str] = None, ve_cutoff: int = 50, dense: bool = True, 
                 exclude_embeddings: bool = True,
                 max_d_model: Optional[int] = None,
                 max_layers: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            vocab_size: Vocabulary size for language models
            max_len: Maximum sequence length
            device: Device to create models on
            ve_cutoff: Virtual edge cutoff for graph construction
            dense: Whether to use dense graphs
            max_d_model: Maximum d_model for GPT Encoder/Mini GPT variants (filters large models)
            max_layers: Maximum layers for GPT Encoder/Mini GPT variants (filters deep models)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ve_cutoff = ve_cutoff
        self.dense = dense
        self.exclude_embeddings = exclude_embeddings
        self.configs = ghn_training_variants(vocab_size=vocab_size, max_len=max_len,
                                            max_d_model=max_d_model,
                                            max_layers=max_layers)

    def __len__(self) -> int:
        """Return the number of model configurations."""
        return len(self.configs)

    def __getitem__(self, idx: int):
        """
        Get a model configuration by index (LAZY LOADING).
        
        Models are created on-demand when this method is called, not pre-loaded.
        This ensures memory efficiency when dealing with large datasets.
        
        Args:
            idx: Index of the configuration
            
        Returns:
            Tuple of (model, graph, meta)
        """
        cfg = self.configs[idx].copy()
        name = cfg.pop("name")
        model_type = cfg.pop("model_type", "tiny_transformer")
        
        # LAZY LOADING: Create model on-demand (not pre-loaded in __init__)
        # For multiprocessing workers, create on CPU to avoid CUDA OOM
        # Models will be moved to GPU in the main process
        worker_device = 'cpu' if self.device == 'cuda' else self.device
        model = create_model_from_config(model_type, cfg, worker_device)
        graph = Graph(model, ve_cutoff=self.ve_cutoff, dense=self.dense, verbose=False, 
                     exclude_embeddings=self.exclude_embeddings)
        meta = {"name": name, "model_type": model_type, **cfg}
        
        return model, graph, meta


# =============================================================================
# DataLoader Functions
# =============================================================================

def _collate_ghn_models(items, dense=True):
    """
    Collate function to batch models, graphs, and metadata.
    Must be a module-level function (not nested) for multiprocessing pickling.
    """
    models, graphs, metas = zip(*items)
    gb = GraphBatch(list(graphs), dense=dense)
    return list(models), gb, list(metas)

def build_ghn_variants_dataloader(
    batch_size: int = 2,
    vocab_size: int = 50257,
    max_len: int = 1024,
    device: Optional[str] = None,
    num_workers: int = 0,
    ve_cutoff: int = 50,
    dense: bool = True,
    shuffle: bool = True,
    exclude_embeddings: bool = True,
    max_d_model: Optional[int] = None,
    max_layers: Optional[int] = None,
):
    """
    Build a DataLoader for GHN training with comprehensive LM variants.
    
    Args:
        batch_size: Number of models per batch
        vocab_size: Vocabulary size for language models
        max_len: Maximum sequence length
        device: Device to create models on
        num_workers: Number of worker processes for data loading
        ve_cutoff: Virtual edge cutoff for graph construction
        dense: Whether to use dense graphs
        shuffle: Whether to shuffle the dataset
        exclude_embeddings: Whether to exclude embedding layers from GHN prediction
        max_d_model: Maximum d_model for GPT Encoder/Mini GPT variants (None = no limit)
                    Suggested: 512 for <40GB memory, 768 for <80GB
        max_layers: Maximum layers for GPT Encoder/Mini GPT variants (None = no limit)
                   Suggested: 8 for <40GB memory, 12 for <80GB
        
    Returns:
        Tuple of (DataLoader, configs) where DataLoader yields (models, graph_batch, metas)
    """
    dataset = GHNLMVariantsDataset(
        vocab_size=vocab_size, 
        max_len=max_len, 
        device=device, 
        ve_cutoff=ve_cutoff, 
        dense=dense,
        exclude_embeddings=exclude_embeddings,
        max_d_model=max_d_model,
        max_layers=max_layers
    )

    # Create a partial function with dense parameter for pickling compatibility
    # Using functools.partial to bind the dense parameter
    from functools import partial
    collate_fn = partial(_collate_ghn_models, dense=dense)

    # Use 'spawn' multiprocessing context for CUDA compatibility
    # This is required when num_workers > 0 and models are moved to CUDA in workers
    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context('spawn')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # Don't keep workers alive between epochs (saves memory)
        multiprocessing_context=mp_context,  # Use 'spawn' for CUDA compatibility
    )
    return loader, dataset.configs
