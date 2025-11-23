#!/usr/bin/env python3
"""
Train a single language model on WikiText-2 dataset using configuration files.

Supports multiple initialization methods:
- default: PyTorch default initialization
- he: He (Kaiming) initialization
- xavier: Xavier (Glorot) initialization
- ghn: GHN-predicted initialization

Usage:
    python train_lm.py --config LM/configs/benchmark_1_tiny.yaml
    python train_lm.py --config LM/configs/benchmark_2_small.yaml --init_method he
    python train_lm.py --config LM/configs/benchmark_2_small.yaml --init_method ghn --ghn_checkpoint Experiment/20917896/best_model.pt
    python train_lm.py --list_configs
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from LM import Trainer
from LM.create_model import create_model
from Dataloader.wikitext2_loader import build_wikitext2
from Dataloader.config_loader import load_config_file, list_benchmark_configs


def initialize_with_he(model):
    """Initialize model parameters using He (Kaiming) initialization."""
    print("🏗️  Initializing model with He (Kaiming) initialization...")
    
    def init_he_weights(module):
        """Apply He initialization to a module."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    model.apply(init_he_weights)
    print("   ✅ He initialization complete!")
    return model


def initialize_with_xavier(model):
    """Initialize model parameters using Xavier (Glorot) initialization."""
    print("🏗️  Initializing model with Xavier (Glorot) initialization...")
    
    def init_xavier_weights(module):
        """Apply Xavier initialization to a module."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    model.apply(init_xavier_weights)
    print("   ✅ Xavier initialization complete!")
    return model


def initialize_with_ghn(model, ghn, device):
    """Initialize model parameters using GHN predictions."""
    print("🏗️  Initializing model with GHN predictions...")
    ghn.eval()
    if device is not None:
        model = model.to(device)
    with torch.no_grad():
        model = ghn(model, keep_grads=False, bn_track_running_stats=True, reduce_graph=False)
    print("   ✅ GHN initialization complete!")
    return model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a language model on WikiText-2 using config files")
    
    # Configuration file option
    parser.add_argument("--config", type=str, required=False,
                       help="Path to configuration file (YAML format)")
    parser.add_argument("--list_configs", action="store_true",
                       help="List all available benchmark configurations and exit")
    parser.add_argument("--init_method", type=str, default="default",
                       choices=["default", "he", "xavier", "ghn"],
                       help="Initialization method: default, he, xavier, or ghn (default: default)")
    parser.add_argument("--ghn_checkpoint", type=str, default=None,
                       help="Path to GHN checkpoint file (required if init_method=ghn)")
    parser.add_argument("--save_ghn_init", action="store_true",
                       help="Save the GHN-initialized model before training (only for ghn init)")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("Available benchmark configurations:")
        configs = list_benchmark_configs()
        for config in configs:
            print(f"  - {config}")
        return
    
    # Check if config is provided when not listing
    if not args.config:
        parser.error("--config is required when not using --list_configs")
    
    # Validate GHN checkpoint if needed
    if args.init_method == "ghn" and not args.ghn_checkpoint:
        parser.error("--ghn_checkpoint is required when --init_method=ghn")
    
    # Load configuration from file
    print(f"📋 Loading configuration from: {args.config}")
    try:
        model_config, training_config, data_config = load_config_file(args.config)
        print(f"   Model: {model_config.model_type}")
        print(f"   Epochs: {training_config.epochs}")
        print(f"   Batch size: {training_config.batch_size}")
        print(f"   Sequence length: {data_config.seq_len}")
        print(f"   Initialization method: {args.init_method}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return
    
    # Setup device from config
    device = torch.device(training_config.device)
    
    # Load GHN model if needed
    ghn = None
    if args.init_method == "ghn":
        print(f"\n🤖 Loading GHN model from: {args.ghn_checkpoint}")
        try:
            from GHN import from_pretrained
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
    
    # Initialize model based on method
    if args.init_method == "he":
        model = initialize_with_he(model)
    elif args.init_method == "xavier":
        model = initialize_with_xavier(model)
    elif args.init_method == "ghn":
        model = initialize_with_ghn(model, ghn, device)
    # default: use PyTorch default initialization (no action needed)
    
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
    if args.save_ghn_init and args.init_method == "ghn":
        main_save_path = f"ghn_init_{model_config.model_type}_{int(time.time())}.pt"
        torch.save(model.state_dict(), main_save_path)
        print(f"💾 GHN-initialized model saved to: {main_save_path}")
        
        exp_save_path = os.path.join(trainer.job_experiment_dir, "zero_epoch.pt")
        torch.save(model.state_dict(), exp_save_path)
        print(f"💾 GHN-initialized model also saved to: {exp_save_path}")
    
    # Update config with actual vocab size
    trainer.update_config_vocab_size(data['vocab_size'])
    
    # Add initialization info to trainer
    trainer.init_method = args.init_method
    if args.init_method == "ghn":
        trainer.ghn_initialized = True
        trainer.ghn_checkpoint = args.ghn_checkpoint
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
