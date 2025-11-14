"""
Language Model Architecture Loader for GHN-3 Training.

This module provides comprehensive language model architecture datasets for training
Graph HyperNetworks (GHN-3) on language modeling tasks. It supports:

- HuggingFace Transformers models (GPT-2 with extensive parameter variations)
- Comprehensive parameter space exploration
- Real architectural differences preserved through Transformers library
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
from LM import TransformersLM


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_model_parameters(d_model: int, n_layers: int, n_heads: int, d_ff: int, 
                               vocab_size: int, max_seq_len: int) -> int:
    """
    Calculate the approximate number of parameters in a GPT-2 style transformer model.
    
    Args:
        d_model: Model dimension (embedding size)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        
    Returns:
        Total number of parameters (approximate)
    """
    # Word token embeddings
    wte_params = vocab_size * d_model
    
    # Position embeddings
    wpe_params = max_seq_len * d_model
    
    # Per transformer layer
    # Attention: QKV projection (3*d_model*d_model) + output projection (d_model*d_model)
    attn_qkv_params = 3 * d_model * d_model
    attn_out_params = d_model * d_model
    
    # Layer norms: 2 per layer, each has 2*d_model params (weight + bias)
    ln_params = 2 * (2 * d_model)
    
    # MLP: feed-forward up and down projections
    mlp_up_params = d_model * d_ff
    mlp_down_params = d_ff * d_model
    
    params_per_layer = attn_qkv_params + attn_out_params + ln_params + mlp_up_params + mlp_down_params
    total_transformer_params = n_layers * params_per_layer
    
    # Final layer norm
    ln_f_params = 2 * d_model
    
    # LM head (no bias in GPT-2)
    lm_head_params = d_model * vocab_size
    
    total_params = wte_params + wpe_params + total_transformer_params + ln_f_params + lm_head_params
    
    return total_params


def create_model_from_config(model_type: str, config: Dict, device: str = 'cpu'):
    """
    Factory function to create different types of language models.
    
    Args:
        model_type: Type of model ('transformers')
        config: Model configuration dictionary
        device: Device to create model on
    
    Returns:
        Instantiated model on specified device
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    if model_type != 'transformers':
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: transformers")
    
    config_copy = config.copy()
    
    # Remove metadata fields that shouldn't be passed to model constructor
    metadata_fields = ['name', 'model_type']
    for field in metadata_fields:
        config_copy.pop(field, None)
    
    # Map n_heads (plural) to n_head (singular) for TransformersLM compatibility
    if 'n_heads' in config_copy and 'n_head' not in config_copy:
        config_copy['n_head'] = config_copy.pop('n_heads')
    
    # Map n_layers to n_layer for TransformersLM compatibility
    if 'n_layers' in config_copy and 'n_layer' not in config_copy:
        config_copy['n_layer'] = config_copy.pop('n_layers')
    
    # Validate d_model and n_head before creating model
    if 'd_model' in config_copy and 'n_head' in config_copy:
        d_model = config_copy['d_model']
        n_head = config_copy['n_head']
        if d_model % n_head != 0:
            raise ValueError(
                f"`d_model` must be divisible by `n_head` "
                f"(got `d_model`: {d_model} and `n_head`: {n_head}). "
                f"Please ensure d_model % n_head == 0."
            )
    
    # Create HuggingFace Transformers model
    model = TransformersLM(**config_copy)
    return model.to(device)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "LMVariantsDataset", 
    "build_lm_variants_dataloader", 
    "lm_training_variants",
    "create_model_from_config"
]


# =============================================================================
# Model Configuration Generation
# =============================================================================

def lm_training_variants(vocab_size: int = 50257, max_len: int = 1024, 
                         max_params: Optional[int] = None) -> List[Dict]:
    """
    Generate comprehensive language model configurations for training.
    
    Creates configurations for HuggingFace Transformers models:
    - Open Source GPT models: GPT-2, GPT-Neo, GPT-J, Mistral, MPT (real architectures)
    
    Args:
        vocab_size: Vocabulary size for language models
        max_len: Maximum sequence length
        max_params: Maximum number of parameters (filters out larger models).
                   Default: None (no filtering). Recommended: 4e9 (4B) or 5e9 (5B) for 40GB GPU.
        
    Returns:
        List of model configuration dictionaries
    """
    variants: List[Dict] = []
    
    # Generate Open Source GPT variants
    variants.extend(_generate_oss_gpt_variants(max_len, vocab_size, max_params=max_params))
    
    return variants


def _generate_oss_gpt_variants(max_len: int, vocab_size: int, 
                                max_params: Optional[int] = None) -> List[Dict]:
    """
    Generate Open Source GPT model variants using HuggingFace Transformers.
    
    Generates comprehensive configurations for GPT-2 model family with
    parameter variations to explore the full architectural space.
    
    Currently supports:
    - GPT-2: gpt2 (with parameter variations)
    
    Args:
        max_len: Maximum sequence length
        vocab_size: Vocabulary size
        max_params: Maximum number of parameters (filters out larger models).
                   Default: None (no filtering). Recommended: 4e9 (4B) or 5e9 (5B) for 40GB GPU.
        
    Returns:
        List of OSS GPT model configuration dictionaries
    """
    variants = []
    
    # Define model families and their base configurations
    model_families = {
        'gpt2': {
            'base_models': ['gpt2'],  # Use single base model, vary parameters
            # Expanded ranges to generate at least 130K variants
            'd_models': list(range(128, 2049, 32)) + list(range(2048, 4097, 64)),  # 128-4096 with different step sizes
            'n_layers_list': list(range(1, 65)),  # 1-64 layers
            'n_heads_list': list(range(1, 65)),  # 1-64 heads (will be filtered to valid divisors)
            'd_ff_ratios': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],  # Various FF ratios
            'max_seq_len': 1024,
        },
        # 'gpt-neo': {
        #     'base_models': ['EleutherAI/gpt-neo-125M'],  # Use single base model, vary parameters
        #     'd_models': [768, 2048, 2560],
        #     'n_layers_list': [12, 24, 32],
        #     'n_heads_list': [12, 16, 20],
        #     'd_ff_ratios': [4.0],  # GPT-Neo uses 4x d_model for d_ff
        #     'max_seq_len': 2048,
        # },
        # 'gpt-j': {
        #     'base_models': ['EleutherAI/gpt-j-6B'],
        #     'd_models': [4096],
        #     'n_layers_list': [28],
        #     'n_heads_list': [16],
        #     'd_ff_ratios': [4.0],  # GPT-J uses 4x d_model for d_ff
        #     'max_seq_len': 2048,
        # },
        # 'mistral': {
        #     'base_models': ['mistralai/Mistral-7B-v0.1'],
        #     'd_models': [4096],
        #     'n_layers_list': [32],
        #     'n_heads_list': [32],
        #     'd_ff_ratios': [3.5],  # Mistral uses ~3.5x d_model for d_ff
        #     'max_seq_len': 32768,
        # },
        # 'mpt': {
        #     'base_models': ['mosaicml/mpt-7b'],
        #     'd_models': [4096],
        #     'n_layers_list': [32],
        #     'n_heads_list': [32],
        #     'd_ff_ratios': [3.5],  # MPT uses ~3.5x d_model for d_ff
        #     'max_seq_len': 2048,
        # },
    }
    
    # Generate variants for each model family
    for family_name, family_config in model_families.items():
        base_models = family_config['base_models']
        d_models = family_config['d_models']
        n_layers_list = family_config['n_layers_list']
        n_heads_list = family_config['n_heads_list']
        d_ff_ratios = family_config['d_ff_ratios']
        family_max_seq_len = family_config['max_seq_len']
        
        # Generate all combinations
        for base_model in base_models:
            for d_model in d_models:
                # Get valid heads that divide d_model
                # Only include heads where d_model is divisible by n_heads
                valid_heads = [h for h in n_heads_list if d_model % h == 0]
                if not valid_heads:
                    # Skip this d_model if no valid heads exist
                    continue
                
                # Double-check: ensure all valid_heads actually divide d_model
                valid_heads = [h for h in valid_heads if d_model % h == 0]
                
                for n_heads in valid_heads:
                    for n_layers in n_layers_list:
                        for d_ff_ratio in d_ff_ratios:
                            d_ff = int(d_model * d_ff_ratio)
                            
                            # Calculate model parameters and filter if max_params is set
                            if max_params is not None:
                                num_params = calculate_model_parameters(
                                    d_model=d_model,
                                    n_layers=n_layers,
                                    n_heads=n_heads,
                                    d_ff=d_ff,
                                    vocab_size=vocab_size,
                                    max_seq_len=min(max_len, family_max_seq_len)
                                )
                                # Skip models that exceed the maximum parameter count
                                if num_params > max_params:
                                    continue
                            
                            # Create variant name
                            variant_name = f"{family_name}-{d_model}d-{n_layers}L-h{n_heads}-ff{d_ff}"
                            
                            variant = {
                                'name': variant_name,
                                'model_type': 'transformers',
                                'model_name': base_model,
                                'vocab_size': vocab_size,
                                'd_model': d_model,
                                'n_layers': n_layers,
                                'n_heads': n_heads,
                                'd_ff': d_ff,
                                'max_len': min(max_len, family_max_seq_len),
                                'p_drop': 0.1,
                                'tie_weights': False,
                            }
                            variants.append(variant)
    
    return variants


# =============================================================================
# Dataset Classes
# =============================================================================

class LMVariantsDataset(Dataset):
    """
    Comprehensive dataset of language model variants for training.
    
    This dataset provides access to HuggingFace Transformers model architectures.
    Each item returns (model, graph, meta) where:
    - model: Instantiated language model
    - graph: Graph representation for GHN-3
    - meta: Metadata including configuration and model type
    
    IMPORTANT: This dataset uses LAZY LOADING - models are created on-demand
    in __getitem__() when requested, not pre-loaded in __init__(). This ensures
    memory efficiency when dealing with large datasets.
    Only model configurations (dictionaries) are stored in memory.
    """

    def __init__(self, vocab_size: int = 50257, max_len: int = 1024, 
                 device: Optional[str] = None, ve_cutoff: int = 50, dense: bool = True, 
                 exclude_embeddings: bool = True, max_params: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            vocab_size: Vocabulary size for language models
            max_len: Maximum sequence length
            device: Device to create models on
            ve_cutoff: Virtual edge cutoff for graph construction
            dense: Whether to use dense graphs
            exclude_embeddings: Whether to exclude embedding layers from GHN prediction
            max_params: Maximum number of parameters (filters out larger models).
                       Default: None (no filtering). Recommended: 4e9 (4B) or 5e9 (5B) for 40GB GPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ve_cutoff = ve_cutoff
        self.dense = dense
        self.exclude_embeddings = exclude_embeddings
        self.configs = lm_training_variants(vocab_size=vocab_size, max_len=max_len, max_params=max_params)

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

def build_lm_variants_dataloader(
    batch_size: int = 2,
    vocab_size: int = 50257,
    max_len: int = 1024,
    device: Optional[str] = None,
    num_workers: int = 0,
    ve_cutoff: int = 50,
    dense: bool = True,
    shuffle: bool = True,
    exclude_embeddings: bool = True,
    max_params: Optional[int] = None,
):
    """
    Build a DataLoader for training with comprehensive LM variants.
    
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
        max_params: Maximum number of parameters (filters out larger models).
                   Default: None (no filtering). Recommended: 4e9 (4B) or 5e9 (5B) for 40GB GPU.
        
    Returns:
        Tuple of (DataLoader, configs) where DataLoader yields (models, graph_batch, metas)
    """
    dataset = LMVariantsDataset(
        vocab_size=vocab_size, 
        max_len=max_len, 
        device=device, 
        ve_cutoff=ve_cutoff, 
        dense=dense,
        exclude_embeddings=exclude_embeddings,
        max_params=max_params
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
