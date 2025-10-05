"""
Model Loader for GHN Training

This module creates diverse configurations for the existing model types:
- RNN, LSTM, GRU, GPT Encoder, Mini GPT

It generates many different architecture variants by varying:
- Hidden dimensions (d_model)
- Number of layers
- Number of heads (for transformers)
- Feed-forward dimensions
- Other hyperparameters
"""

import torch
import random
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

# Import your existing models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import (
    RNNLanguageModel, RNNConfig,
    LSTMLanguageModel, LSTMConfig,
    GRULanguageModel, GRUConfig,
    GPTEncoderLayerLM, GPTEncoderConfig,
    GPTDecoderLM, MiniGPTConfig
)


class ModelConfigGenerator:
    """Generates diverse model configurations for different architecture types."""
    
    def __init__(self, vocab_size: int = 50257, max_seq_len: int = 512, seed: int = 42):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
    
    def _get_valid_heads(self, d_model: int, requested_heads: int) -> int:
        """Ensure number of heads divides d_model."""
        if d_model % requested_heads == 0:
            return requested_heads
        
        # Try common head counts
        for heads in [16, 12, 8, 6, 4, 3, 2]:
            if heads <= d_model and d_model % heads == 0:
                return heads
        return 2
    
    def generate_rnn_configs(self) -> List[Dict]:
        """Generate ALL possible RNN model configurations."""
        configs = []
        
        # Define comprehensive parameter ranges
        d_models = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
        n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32]
        dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        tie_weights_options = [True, False]
        
        config_id = 1
        for d_model in d_models:
            for n_layer in n_layers:
                for dropout in dropouts:
                    for tie_weights in tie_weights_options:
                        config = {
                            "model_type": "rnn",
                            "name": f"rnn-{config_id:03d}-{d_model}d-{n_layer}L-d{dropout:.2f}-t{tie_weights}",
                            "vocab_size": self.vocab_size,
                            "d_model": d_model,
                            "n_layer": n_layer,
                            "max_seq_len": self.max_seq_len,
                            "p_drop": dropout,
                            "tie_weights": tie_weights
                        }
                        configs.append(config)
                        config_id += 1
        
        return configs
    
    def generate_lstm_configs(self) -> List[Dict]:
        """Generate ALL possible LSTM model configurations."""
        configs = []
        
        # Define comprehensive parameter ranges
        d_models = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
        n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32]
        dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        tie_weights_options = [True, False]
        
        config_id = 1
        for d_model in d_models:
            for n_layer in n_layers:
                for dropout in dropouts:
                    for tie_weights in tie_weights_options:
                        config = {
                            "model_type": "lstm",
                            "name": f"lstm-{config_id:03d}-{d_model}d-{n_layer}L-d{dropout:.2f}-t{tie_weights}",
                            "vocab_size": self.vocab_size,
                            "d_model": d_model,
                            "n_layer": n_layer,
                            "max_seq_len": self.max_seq_len,
                            "p_drop": dropout,
                            "tie_weights": tie_weights
                        }
                        configs.append(config)
                        config_id += 1
        
        return configs
    
    def generate_gru_configs(self) -> List[Dict]:
        """Generate ALL possible GRU model configurations."""
        configs = []
        
        # Define comprehensive parameter ranges
        d_models = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
        n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32]
        dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        tie_weights_options = [True, False]
        
        config_id = 1
        for d_model in d_models:
            for n_layer in n_layers:
                for dropout in dropouts:
                    for tie_weights in tie_weights_options:
                        config = {
                            "model_type": "gru",
                            "name": f"gru-{config_id:03d}-{d_model}d-{n_layer}L-d{dropout:.2f}-t{tie_weights}",
                            "vocab_size": self.vocab_size,
                            "d_model": d_model,
                            "n_layer": n_layer,
                            "max_seq_len": self.max_seq_len,
                            "p_drop": dropout,
                            "tie_weights": tie_weights
                        }
                        configs.append(config)
                        config_id += 1
        
        return configs
    
    def generate_gpt_encoder_configs(self) -> List[Dict]:
        """Generate ALL possible GPT Encoder model configurations."""
        configs = []
        
        # Define comprehensive parameter ranges
        d_models = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
        n_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]
        n_heads_options = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        d_ff_ratios = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]  # d_ff = d_model * ratio
        dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        tie_weights_options = [True, False]
        
        config_id = 1
        for d_model in d_models:
            for n_layer in n_layers:
                for n_heads in n_heads_options:
                    # Only include if heads divide d_model
                    if d_model % n_heads != 0:
                        continue
                    for d_ff_ratio in d_ff_ratios:
                        d_ff = d_model * d_ff_ratio
                        for dropout in dropouts:
                            for tie_weights in tie_weights_options:
                                config = {
                                    "model_type": "gpt_encoder",
                                    "name": f"gpt-enc-{config_id:03d}-{d_model}d-{n_layer}L-{n_heads}h-ff{d_ff_ratio}-d{dropout:.2f}-t{tie_weights}",
                                    "vocab_size": self.vocab_size,
                                    "d_model": d_model,
                                    "n_layer": n_layer,
                                    "n_head": n_heads,
                                    "d_ff": d_ff,
                                    "max_seq_len": self.max_seq_len,
                                    "p_drop": dropout,
                                    "tie_weights": tie_weights
                                }
                                configs.append(config)
                                config_id += 1
        
        return configs
    
    def generate_mini_gpt_configs(self) -> List[Dict]:
        """Generate ALL possible Mini GPT model configurations."""
        configs = []
        
        # Define comprehensive parameter ranges
        d_models = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
        n_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]
        n_heads_options = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        d_ff_ratios = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]  # d_ff = d_model * ratio
        dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        tie_weights_options = [True, False]
        
        config_id = 1
        for d_model in d_models:
            for n_layer in n_layers:
                for n_heads in n_heads_options:
                    # Only include if heads divide d_model
                    if d_model % n_heads != 0:
                        continue
                    for d_ff_ratio in d_ff_ratios:
                        d_ff = d_model * d_ff_ratio
                        for dropout in dropouts:
                            for tie_weights in tie_weights_options:
                                config = {
                                    "model_type": "mini_gpt",
                                    "name": f"mini-gpt-{config_id:03d}-{d_model}d-{n_layer}L-{n_heads}h-ff{d_ff_ratio}-d{dropout:.2f}-t{tie_weights}",
                                    "vocab_size": self.vocab_size,
                                    "d_model": d_model,
                                    "n_layer": n_layer,
                                    "n_head": n_heads,
                                    "d_ff": d_ff,
                                    "max_seq_len": self.max_seq_len,
                                    "p_drop": dropout,
                                    "tie_weights": tie_weights
                                }
                                configs.append(config)
                                config_id += 1
        
        return configs
    
    def generate_all_configs(self) -> List[Dict]:
        """Generate ALL possible model configurations."""
        all_configs = []
        
        rnn_configs = self.generate_rnn_configs()
        all_configs.extend(rnn_configs)
        lstm_configs = self.generate_lstm_configs()
        all_configs.extend(lstm_configs)
        gru_configs = self.generate_gru_configs()
        all_configs.extend(gru_configs)
        gpt_encoder_configs = self.generate_gpt_encoder_configs()
        all_configs.extend(gpt_encoder_configs)
        mini_gpt_configs = self.generate_mini_gpt_configs()
        all_configs.extend(mini_gpt_configs)
        return all_configs
    
    def generate_reasonable_configs(self) -> List[Dict]:
        """Generate a reasonable subset of configurations for practical use."""
        all_configs = []
        
        
        # For RNN/LSTM/GRU: Use a subset of parameters
        rnn_d_models = [64, 128, 256, 384, 512, 768, 1024]
        rnn_n_layers = [1, 2, 3, 4, 6, 8, 12, 16]
        rnn_dropouts = [0.0, 0.1, 0.2, 0.3]
        rnn_tie_weights = [True, False]
        
        config_id = 1
        for model_type in ["rnn", "lstm", "gru"]:
            for d_model in rnn_d_models:
                for n_layer in rnn_n_layers:
                    for dropout in rnn_dropouts:
                        for tie_weights in rnn_tie_weights:
                            config = {
                                "model_type": model_type,
                                "name": f"{model_type}-{config_id:03d}-{d_model}d-{n_layer}L-d{dropout:.1f}-t{tie_weights}",
                                "vocab_size": self.vocab_size,
                                "d_model": d_model,
                                "n_layer": n_layer,
                                "max_seq_len": self.max_seq_len,
                                "p_drop": dropout,
                                "tie_weights": tie_weights
                            }
                            all_configs.append(config)
                            config_id += 1
        
        # For Transformer models: Use a subset of parameters
        transformer_d_models = [64, 128, 256, 384, 512, 768, 1024]
        transformer_n_layers = [2, 3, 4, 6, 8, 12, 16, 24]
        transformer_n_heads = [2, 4, 6, 8, 12, 16]
        transformer_d_ff_ratios = [2, 3, 4, 6, 8]
        transformer_dropouts = [0.0, 0.1, 0.2]
        transformer_tie_weights = [True, False]
        
        for model_type in ["gpt_encoder", "mini_gpt"]:
            for d_model in transformer_d_models:
                for n_layer in transformer_n_layers:
                    for n_heads in transformer_n_heads:
                        if d_model % n_heads != 0:
                            continue
                        for d_ff_ratio in transformer_d_ff_ratios:
                            d_ff = d_model * d_ff_ratio
                            for dropout in transformer_dropouts:
                                for tie_weights in transformer_tie_weights:
                                    config = {
                                        "model_type": model_type,
                                        "name": f"{model_type.replace('_', '-')}-{config_id:03d}-{d_model}d-{n_layer}L-{n_heads}h-ff{d_ff_ratio}-d{dropout:.1f}-t{tie_weights}",
                                        "vocab_size": self.vocab_size,
                                        "d_model": d_model,
                                        "n_layer": n_layer,
                                        "n_head": n_heads,
                                        "d_ff": d_ff,
                                        "max_seq_len": self.max_seq_len,
                                        "p_drop": dropout,
                                        "tie_weights": tie_weights
                                    }
                                    all_configs.append(config)
                                    config_id += 1
        
        return all_configs


class ModelDataset(Dataset):
    """Dataset that creates models from configurations."""
    
    def __init__(self, configs: List[Dict], device: Optional[str] = None):
        self.configs = configs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def __len__(self) -> int:
        return len(self.configs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.nn.Module, Dict]:
        """Create and return a model with its configuration."""
        config = self.configs[idx].copy()
        model_type = config.pop("model_type")
        name = config.pop("name")
        
        # Create model based on type
        if model_type == "rnn":
            model_config = RNNConfig(**config)
            model = RNNLanguageModel(model_config)
        elif model_type == "lstm":
            model_config = LSTMConfig(**config)
            model = LSTMLanguageModel(model_config)
        elif model_type == "gru":
            model_config = GRUConfig(**config)
            model = GRULanguageModel(model_config)
        elif model_type == "gpt_encoder":
            model_config = GPTEncoderConfig(**config)
            model = GPTEncoderLayerLM(model_config)
        elif model_type == "mini_gpt":
            model_config = MiniGPTConfig(**config)
            model = GPTDecoderLM(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to device
        model = model.to(self.device)
        
        # Add metadata
        metadata = {
            "name": name,
            "model_type": model_type,
            "config": config,
            "num_params": sum(p.numel() for p in model.parameters())
        }
        
        return model, metadata


def create_model_dataloader(
    vocab_size: int = 50257,
    max_seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    device: Optional[str] = None,
    seed: int = 42,
    use_all_configs: bool = False
) -> Tuple[DataLoader, List[Dict]]:
    """
    Create a DataLoader with model configurations.
    
    Args:
        use_all_configs: If True, generates ALL possible configurations (3M+).
                        If False, generates a reasonable subset (~10K).
    
    Returns:
        DataLoader: Loader that yields (model, metadata) pairs
        List[Dict]: All configurations used
    """
    
    # Generate configurations
    generator = ModelConfigGenerator(vocab_size, max_seq_len, seed)
    if use_all_configs:
        configs = generator.generate_all_configs()
    else:
        configs = generator.generate_reasonable_configs()
    
    # Create dataset
    dataset = ModelDataset(configs, device)
    
    # Create dataloader
    def collate_fn(batch):
        models, metadatas = zip(*batch)
        return list(models), list(metadatas)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, configs


def create_reasonable_model_dataloader(
    vocab_size: int = 50257,
    max_seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    device: Optional[str] = None,
    seed: int = 42
) -> Tuple[DataLoader, List[Dict]]:
    """
    Create a DataLoader with a reasonable subset of model configurations (~10K).
    
    Returns:
        DataLoader: Loader that yields (model, metadata) pairs
        List[Dict]: All configurations used
    """
    return create_model_dataloader(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        use_all_configs=False
    )


def create_full_model_dataloader(
    vocab_size: int = 50257,
    max_seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    device: Optional[str] = None,
    seed: int = 42
) -> Tuple[DataLoader, List[Dict]]:
    """
    Create a DataLoader with ALL possible model configurations (3M+).
    
    Returns:
        DataLoader: Loader that yields (model, metadata) pairs
        List[Dict]: All configurations used
    """
    return create_model_dataloader(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        use_all_configs=True
    )





