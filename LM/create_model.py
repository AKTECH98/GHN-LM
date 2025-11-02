#!/usr/bin/env python3
"""
Create language models from benchmark configuration files.

Usage:
    python create_model.py --config benchmark_1_tiny
    python create_model.py --config benchmark_2_small
    python create_model.py --list_configs
    python create_model.py --config benchmark_1_tiny --save_model
"""

import argparse
import os

import torch

from LM import (
    GPTEncoderLayerLM, GPTEncoderConfig,
    GPTDecoderLM, MiniGPTConfig
)
from Dataloader.config_loader import load_config_file, list_benchmark_configs


def create_model(model_config, vocab_size=50257):
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
    parser = argparse.ArgumentParser(description="Create language models from benchmark configuration files")
    
    # Configuration file option
    parser.add_argument("--config", type=str,
                       help="Benchmark configuration name (e.g., benchmark_1_tiny)")
    parser.add_argument("--list_configs", action="store_true",
                       help="List all available benchmark configurations and exit")
    parser.add_argument("--save_model", action="store_true",
                       help="Save the created model to a file")
    parser.add_argument("--vocab_size", type=int, default=50257,
                       help="Vocabulary size (default: 50257)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to create model on (default: cpu)")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("Available benchmark configurations:")
        configs = list_benchmark_configs()
        for config in configs:
            print(f"  - {config}")
        return
    
    if not args.config:
        print("❌ Error: --config is required. Use --list_configs to see available configs.")
        return
    
    # Load configuration from file
    config_path = f"LM/configs/{args.config}.yaml"
    print(f"📋 Loading configuration from: {config_path}")
    
    try:
        model_config, training_config, data_config = load_config_file(config_path)
        print(f"   Model: {model_config.model_type}")
        print(f"   D_model: {model_config.d_model}")
        print(f"   N_layers: {model_config.n_layer}")
        print(f"   N_heads: {model_config.n_head}")
        print(f"   Max seq len: {model_config.max_seq_len}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return
    
    # Setup device
    device = torch.device(args.device)
    
    # Create model
    print(f"\n🏗️  Creating {model_config.model_type} model...")
    model = create_model(model_config, args.vocab_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model device: {next(model.parameters()).device}")
    
    # Save model if requested
    if args.save_model:
        save_path = f"models/{args.config}_model.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved to: {save_path}")
    
    # Test model with dummy input
    print(f"\n🧪 Testing model with dummy input...")
    model.eval()
    with torch.no_grad():
        # Create dummy input
        batch_size = 2
        seq_len = min(model_config.max_seq_len, 128)  # Use smaller seq_len for testing
        dummy_input = torch.randint(0, args.vocab_size, (batch_size, seq_len)).to(device)
        
        try:
            output = model(dummy_input)
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   ✅ Model forward pass successful!")
        except Exception as e:
            print(f"   ❌ Model forward pass failed: {e}")
    
    print(f"\n✅ Model creation completed successfully!")
    return model


if __name__ == "__main__":
    main()
