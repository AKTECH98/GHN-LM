#!/usr/bin/env python3
"""
Script to run GHN-initialized training with different configurations.

Usage:
    python run_ghn_init_training.py --config benchmark_1_tiny
    python run_ghn_init_training.py --config benchmark_2_small --ghn_checkpoint Experiment/20921440/best_model.pt
    python run_ghn_init_training.py --list_configs
"""

import argparse
import os
import subprocess

def list_available_configs():
    """List available benchmark configurations."""
    config_dir = "LM/configs"
    configs = []
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.yaml') and file.startswith('benchmark_'):
                configs.append(file.replace('.yaml', ''))
    return sorted(configs)

def run_training(config, ghn_checkpoint, save_ghn_init=False, device="cuda"):
    """Run GHN-initialized training with specified configuration."""
    config_path = f"LM/configs/{config}.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    if not os.path.exists(ghn_checkpoint):
        print(f"❌ GHN checkpoint not found: {ghn_checkpoint}")
        return False
    
    print(f"🚀 Starting GHN-initialized training...")
    print(f"   Config: {config}")
    print(f"   GHN Checkpoint: {ghn_checkpoint}")
    print(f"   Device: {device}")
    print(f"   Save GHN init: {save_ghn_init}")
    
    # Build command
    cmd = [
        "python", "train_lm_ghn_init.py",
        "--config", config_path,
        "--ghn_checkpoint", ghn_checkpoint
    ]
    
    if save_ghn_init:
        cmd.append("--save_ghn_init")
    
    print(f"   Command: {' '.join(cmd)}")
    print("="*60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run GHN-initialized language model training")
    
    parser.add_argument("--config", type=str, help="Benchmark configuration name (e.g., benchmark_1_tiny)")
    parser.add_argument("--ghn_checkpoint", type=str, required=True,
                       help="Path to GHN checkpoint file (e.g., Experiment/{job_id}/best_model.pt)")
    parser.add_argument("--save_ghn_init", action="store_true",
                       help="Save the GHN-initialized model before training")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use for training")
    parser.add_argument("--list_configs", action="store_true",
                       help="List available benchmark configurations")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("Available benchmark configurations:")
        configs = list_available_configs()
        for config in configs:
            print(f"  - {config}")
        return
    
    if not args.config:
        parser.error("--config is required. Use --list_configs to see available configs.")
        return
    
    # Run training
    success = run_training(
        config=args.config,
        ghn_checkpoint=args.ghn_checkpoint,
        save_ghn_init=args.save_ghn_init,
        device=args.device
    )
    
    if success:
        print("\n🎉 GHN-initialized training completed successfully!")
    else:
        print("\n💥 GHN-initialized training failed!")
        exit(1)

if __name__ == "__main__":
    main()

