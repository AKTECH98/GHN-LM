"""
Simple Configuration Loader for Individual Config Files
Loads YAML config files with model, training, and data sections
"""

import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    model_type: str
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    max_seq_len: int
    p_drop: float


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    save_interval: int
    eval_interval: int
    log_interval: int
    device: str
    mixed_precision: bool
    gradient_accumulation_steps: int
    seed: int = None
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    seq_len: int
    num_workers: int


def load_config_file(config_path: str) -> tuple[ModelConfig, TrainingConfig, DataConfig]:
    """
    Load a single config file and return model, training, and data configs
    
    Args:
        config_path: Path to the YAML config file
    
    Returns:
        Tuple of (ModelConfig, TrainingConfig, DataConfig)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Validate required sections
    required_sections = ['model', 'training', 'data']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    # Create model config
    model_dict = config_dict['model']
    required_model_fields = ['model_type', 'vocab_size', 'd_model', 'n_layer', 'n_head', 'd_ff', 'max_seq_len', 'p_drop']
    for field in required_model_fields:
        if field not in model_dict:
            raise ValueError(f"Missing required field '{field}' in model section")
    
    model_config = ModelConfig(**model_dict)
    
    # Create training config
    training_dict = config_dict['training']
    required_training_fields = ['epochs', 'batch_size', 'learning_rate', 'weight_decay', 'warmup_steps', 
                               'max_grad_norm', 'save_interval', 'eval_interval', 'log_interval', 
                               'device', 'mixed_precision', 'gradient_accumulation_steps']
    for field in required_training_fields:
        if field not in training_dict:
            raise ValueError(f"Missing required field '{field}' in training section")
    
    # Add optional seed parameter
    if 'seed' in training_dict:
        training_dict['seed'] = training_dict['seed']
    else:
        training_dict['seed'] = None
    
    training_config = TrainingConfig(**training_dict)
    
    # Create data config
    data_dict = config_dict['data']
    required_data_fields = ['seq_len', 'num_workers']
    for field in required_data_fields:
        if field not in data_dict:
            raise ValueError(f"Missing required field '{field}' in data section")
    
    data_config = DataConfig(**data_dict)
    
    return model_config, training_config, data_config


def list_benchmark_configs(config_dir: str = "LM/configs") -> list:
    """List all available benchmark config files"""
    benchmark_files = []
    for file in os.listdir(config_dir):
        if file.startswith("benchmark_") and file.endswith(".yaml"):
            benchmark_files.append(file)
    return sorted(benchmark_files)


if __name__ == "__main__":
    # Example usage
    config_dir = "LM/configs"
    benchmark_configs = list_benchmark_configs(config_dir)
    
    print("Available benchmark configurations:")
    for config_file in benchmark_configs:
        print(f"  - {config_file}")
    
    # Load a specific config
    if benchmark_configs:
        config_path = os.path.join(config_dir, benchmark_configs[0])
        try:
            model_config, training_config, data_config = load_config_file(config_path)
            print(f"\nLoaded config from {config_path}:")
            print(f"Model: {model_config.model_type} (d_model={model_config.d_model}, n_layer={model_config.n_layer})")
            print(f"Training: {training_config.epochs} epochs, batch_size={training_config.batch_size}")
            print(f"Data: seq_len={data_config.seq_len}")
        except Exception as e:
            print(f"Error loading config: {e}")
