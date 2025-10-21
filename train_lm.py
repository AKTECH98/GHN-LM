#!/usr/bin/env python3
"""
Train a single language model on WikiText-2 dataset using configuration files.

Usage:
    python train_lm.py --config LM/configs/benchmark_1_tiny.yaml
    python train_lm.py --config LM/configs/benchmark_2_small.yaml
    python train_lm.py --list_configs
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(__file__))

from LM import (
    GPTEncoderLayerLM, GPTEncoderConfig,
    GPTDecoderLM, MiniGPTConfig,
    Trainer
)
from Dataloader.wikitext2_loader import build_wikitext2
from simple_logger import SimpleLogger
from Dataloader.config_loader import load_config_file, list_benchmark_configs


def create_model(model_config, vocab_size):
    """Create a model based on the model config."""
    if model_config.model_type == "gpt_encoder":
        config = GPTEncoderConfig(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            d_ff=model_config.d_ff,
            max_seq_len=model_config.max_seq_len,
            p_drop=model_config.p_drop
        )
        return GPTEncoderLayerLM(config)
    
    elif model_config.model_type == "mini_gpt":
        config = MiniGPTConfig(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            d_ff=model_config.d_ff,
            max_seq_len=model_config.max_seq_len,
            p_drop=model_config.p_drop
        )
        return GPTDecoderLM(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a language model on WikiText-2 using config files")
    
    # Configuration file option
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file (YAML format)")
    parser.add_argument("--list_configs", action="store_true",
                       help="List all available benchmark configurations and exit")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("Available benchmark configurations:")
        configs = list_benchmark_configs()
        for config in configs:
            print(f"  - {config}")
        return
    
    # Load configuration from file
    print(f"📋 Loading configuration from: {args.config}")
    try:
        model_config, training_config, data_config = load_config_file(args.config)
        print(f"   Model: {model_config.model_type}")
        print(f"   Epochs: {training_config.epochs}")
        print(f"   Batch size: {training_config.batch_size}")
        print(f"   Sequence length: {data_config.seq_len}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return
    
    # Setup device from config
    device = torch.device(training_config.device)
    
    # Load dataset with config parameters
    print(f"\n📚 Loading WikiText-2 dataset...")
    data = build_wikitext2(
        tokenizer_name="gpt2",  # Default tokenizer
        seq_len=data_config.seq_len,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
        cache_dir="./data"
    )
    
    print(f"   Vocab size: {data['vocab_size']}")
    print(f"   Train batches: {len(data['train_loader'])}")
    print(f"   Val batches: {len(data['val_loader'])}")
    
    # Create model with config
    print(f"\n🏗️  Creating {model_config.model_type} model...")
    model = create_model(model_config, data['vocab_size'])
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with config
    trainer = Trainer(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        device=device,
        training_config=training_config,
        model_config=model_config,
        data_config=data_config
    )
    
    # Update logger with actual vocab size
    trainer.update_logger_config(data['vocab_size'])
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
