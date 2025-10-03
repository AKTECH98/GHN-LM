#!/usr/bin/env python3
"""
Simple script to plot training logs from the SimpleLogger.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_training_logs(logs_dir="training_logs", output_dir="plots"):
    """Plot training logs from CSV files."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV log files
    csv_files = []
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(logs_dir, file))
    
    if not csv_files:
        print(f"❌ No CSV log files found in {logs_dir}")
        return
    
    print(f"📊 Found {len(csv_files)} training log files")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, csv_file in enumerate(csv_files):
        # Load data
        df = pd.read_csv(csv_file)
        model_name = os.path.basename(csv_file).split('_')[0]
        color = colors[i % len(colors)]
        
        # Plot 1: Training Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], 
                       label=f'{model_name}', color=color, linewidth=2)
        
        # Plot 2: Validation Loss
        axes[0, 1].plot(df['epoch'], df['val_loss'], 
                       label=f'{model_name}', color=color, linewidth=2)
        
        # Plot 3: Learning Rate
        axes[1, 0].plot(df['epoch'], df['learning_rate'], 
                       label=f'{model_name}', color=color, linewidth=2)
        
        # Plot 4: Both losses together
        axes[1, 1].plot(df['epoch'], df['train_loss'], 
                       label=f'{model_name} (Train)', color=color, linewidth=2, linestyle='-')
        axes[1, 1].plot(df['epoch'], df['val_loss'], 
                       label=f'{model_name} (Val)', color=color, linewidth=2, linestyle='--')
    
    # Set labels and titles
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Learning Rate', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_title('Training vs Validation Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'training_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to {plot_file}")
    
    # Also create individual plots for each model
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        model_name = os.path.basename(csv_file).split('_')[0]
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Losses
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2)
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
        plt.title(f'{model_name} - Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Learning Rate
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['learning_rate'], color='green', linewidth=2)
        plt.title(f'{model_name} - Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Subplot 3: Training Loss only
        plt.subplot(2, 2, 3)
        plt.plot(df['epoch'], df['train_loss'], color='blue', linewidth=2)
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Validation Loss only
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['val_loss'], color='red', linewidth=2)
        plt.title(f'{model_name} - Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        individual_plot = os.path.join(output_dir, f'{model_name}_training.png')
        plt.savefig(individual_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Individual plot saved to {individual_plot}")


def show_training_summary(logs_dir="training_logs"):
    """Show a summary of all training runs."""
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found: {logs_dir}")
        return
    
    print("📊 TRAINING SUMMARY")
    print("="*50)
    
    # Find all JSON log files
    json_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ No JSON log files found")
        return
    
    for json_file in json_files:
        with open(os.path.join(logs_dir, json_file), 'r') as f:
            log_data = json.load(f)
        
        print(f"\n🔬 Model: {log_data['model_name']}")
        print(f"   Timestamp: {log_data['timestamp']}")
        print(f"   Config: {log_data['config']}")
        
        if log_data['epochs']:
            epochs = log_data['epochs']
            print(f"   Total Epochs: {len(epochs)}")
            print(f"   Final Train Loss: {epochs[-1]['train_loss']:.4f}")
            print(f"   Final Val Loss: {epochs[-1]['val_loss']:.4f}")
            print(f"   Best Val Loss: {min(epoch['val_loss'] for epoch in epochs):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument("--logs_dir", default="training_logs", 
                       help="Directory containing training logs")
    parser.add_argument("--output_dir", default="plots", 
                       help="Output directory for plots")
    parser.add_argument("--summary", action="store_true", 
                       help="Show training summary")
    
    args = parser.parse_args()
    
    if args.summary:
        show_training_summary(args.logs_dir)
    else:
        print("🎨 Generating training plots...")
        plot_training_logs(args.logs_dir, args.output_dir)
        print(f"\n✅ Plots saved to {args.output_dir}/")
    
    print(f"\n📁 Available files:")
    print(f"   📊 Logs: {args.logs_dir}/")
    print(f"   📈 Plots: {args.output_dir}/")


if __name__ == "__main__":
    main()
