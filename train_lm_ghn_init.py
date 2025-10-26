#!/usr/bin/env python3
"""
Train a language model initialized with GHN-predicted parameters on WikiText-2 dataset.

Usage:
    python train_lm_ghn_init.py --config LM/configs/benchmark_1_tiny.yaml --ghn_checkpoint Experiment/20917896/best_model.pt
    python train_lm_ghn_init.py --config LM/configs/benchmark_2_small.yaml --ghn_checkpoint Experiment/20917896/best_model.pt
    python train_lm_ghn_init.py --list_configs
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

from LM import Trainer
from LM.create_model import create_model
from Dataloader.wikitext2_loader import build_wikitext2
from Dataloader.config_loader import load_config_file, list_benchmark_configs
from GHN import from_pretrained, Graph


def initialize_with_ghn(model, ghn, device, model_config, vocab_size):
    """Initialize model parameters using GHN predictions."""
    print("🏗️  Initializing model with GHN predictions...")
    
    # Use the GHN model that was already loaded and passed as parameter
    ghn.eval()
    
    model = ghn(model)

    return model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a GHN-initialized language model on WikiText-2")
    
    # Configuration file option
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file (YAML format)")
    parser.add_argument("--ghn_checkpoint", type=str, required=True,
                       help="Path to GHN checkpoint file")
    parser.add_argument("--list_configs", action="store_true",
                       help="List all available benchmark configurations and exit")
    parser.add_argument("--save_ghn_init", action="store_true",
                       help="Save the GHN-initialized model before training")
    
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
    
    # Load GHN model
    print(f"\n🤖 Loading GHN model from: {args.ghn_checkpoint}")
    try:
        ghn = from_pretrained(args.ghn_checkpoint, debug_level=0).to(device)
        print(f"   GHN parameters: {sum(p.numel() for p in ghn.parameters()):,}")
    except Exception as e:
        print(f"❌ Error loading GHN checkpoint: {e}")
        return
    
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
    
    # Initialize model with GHN predictions
    model = initialize_with_ghn(model, ghn, device, model_config, data['vocab_size'])
    
    # Create trainer with config first to get experiment directory
    trainer = Trainer(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        device=device,
        training_config=training_config,
        model_config=model_config,
        data_config=data_config
    )
    
    # Save GHN-initialized model if requested
    if args.save_ghn_init:
        # Save in main directory (original behavior)
        main_save_path = f"ghn_init_{model_config.model_type}_{int(time.time())}.pt"
        torch.save(model.state_dict(), main_save_path)
        print(f"💾 GHN-initialized model saved to: {main_save_path}")
        
        # Also save in experiment directory as zero_epoch.pt
        exp_save_path = os.path.join(trainer.job_experiment_dir, "zero_epoch.pt")
        torch.save(model.state_dict(), exp_save_path)
        print(f"💾 GHN-initialized model also saved to: {exp_save_path}")
    
    # Update config with actual vocab size
    trainer.update_config_vocab_size(data['vocab_size'])
    
    # Add GHN initialization info to trainer
    trainer.ghn_initialized = True
    trainer.ghn_checkpoint = args.ghn_checkpoint
    
    print(f"\n🚀 Starting training with GHN-initialized model...")
    print(f"   Model: {model_config.model_type}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   GHN checkpoint: {args.ghn_checkpoint}")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
