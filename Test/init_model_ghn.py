#!/usr/bin/env python3
"""
Initialize a model from a config file using GHN checkpoint.

Usage:
    python Test/init_model_ghn.py --config LM/configs/benchmark_1_tiny.yaml --ghn_checkpoint GHN_Models/20917896.pt
"""

import argparse
import sys
import os

import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LM.create_model import create_model
from Dataloader.config_loader import load_config_file
from GHN import from_pretrained


def initialize_model_with_ghn(config_path: str, ghn_checkpoint_path: str, device: str = "cpu"):
    """
    Initialize a model from config using GHN checkpoint.
    
    Args:
        config_path: Path to the model configuration YAML file
        ghn_checkpoint_path: Path to the GHN checkpoint file
        device: Device to load models on (default: "cpu")
    
    Returns:
        Model initialized with GHN predictions
    """
    print(f"📋 Loading configuration from: {config_path}")
    try:
        model_config, training_config, data_config = load_config_file(config_path)
        print(f"   Model type: {model_config.model_type}")
        print(f"   D_model: {model_config.d_model}")
        print(f"   N_layers: {model_config.n_layer}")
        print(f"   N_heads: {model_config.n_head}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return None
    
    # Setup device
    device_obj = torch.device(device)
    
    # Load GHN model
    print(f"\n🤖 Loading GHN model from: {ghn_checkpoint_path}")
    try:
        ghn = from_pretrained(ghn_checkpoint_path, debug_level=0).to(device_obj)
        print(f"   GHN parameters: {sum(p.numel() for p in ghn.parameters()):,}")
        ghn.eval()
    except Exception as e:
        print(f"❌ Error loading GHN checkpoint: {e}")
        return None
    
    # Create model from config
    print(f"\n🏗️  Creating {model_config.model_type} model...")
    try:
        model = create_model(model_config, vocab_size=model_config.vocab_size)
        model = model.to(device_obj)
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None
    
    # Initialize model with GHN predictions
    print(f"\n🎯 Initializing model with GHN predictions...")
    try:
        with torch.no_grad():
            model = ghn(model)
        print(f"   ✅ Model successfully initialized with GHN!")
    except Exception as e:
        print(f"❌ Error initializing model with GHN: {e}")
        return None
    
    return model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Initialize a model from config using GHN checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python Test/init_model_ghn.py --config LM/configs/benchmark_1_tiny.yaml --ghn_checkpoint GHN_Models/20917896.pt
    python Test/init_model_ghn.py --config LM/configs/benchmark_2_small.yaml --ghn_checkpoint GHN_Models/20917896.pt --device cuda
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--ghn_checkpoint",
        type=str,
        required=True,
        help="Path to GHN checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models on (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate paths exist
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.ghn_checkpoint):
        print(f"❌ Error: GHN checkpoint file not found: {args.ghn_checkpoint}")
        return 1
    
    # Initialize model
    model = initialize_model_with_ghn(args.config, args.ghn_checkpoint, args.device)
    
    if model is None:
        print("\n❌ Failed to initialize model")
        return 1
    
    print(f"\n✅ Model initialization completed successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {next(model.parameters()).device}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

