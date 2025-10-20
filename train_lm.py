#!/usr/bin/env python3
"""
Train a single language model on WikiText-2 dataset.

This script allows you to train any model from the models folder on the WikiText-2 dataset.
It supports available models: GPT Encoder and Mini GPT.

Usage:
    python train_lm.py --model gpt_encoder --epochs 10 --batch_size 8
    python train_lm.py --model mini_gpt --epochs 5 --d_model 256 --n_layer 4
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
    GPTDecoderLM, MiniGPTConfig
)
from Dataloader.wikitext2_loader import build_wikitext2
from simple_logger import SimpleLogger


class ModelTrainer:
    """Trainer class for language models."""
    
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.1
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Setup logging
        self.writer = SummaryWriter(f"runs/{args.model}_{int(time.time())}")
        
        # Setup simple logger (will be updated with actual vocab_size after data loading)
        config = {
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "vocab_size": "TBD",  # Will be updated after data loading
            "dropout": getattr(args, 'dropout', 0.1)
        }
        self.logger = SimpleLogger(args.model, config, args)
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def update_logger_config(self, vocab_size):
        """Update logger config with actual vocab size from tokenizer."""
        self.logger.log_data["config"]["vocab_size"] = vocab_size
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                # Models that accept targets parameter
                output = self.model(input_ids, targets=labels)
                if len(output) == 3:
                    # Some models return (logits, loss, hidden)
                    logits, loss, hidden = output
                else:
                    # Most models return (logits, loss)
                    logits, loss = output
                
                if loss is None:
                    # If model doesn't compute loss, compute it manually
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                # Models that don't accept targets parameter
                output = self.model(input_ids)
                if len(output) == 3:
                    # Some models return (logits, loss, hidden)
                    logits, loss, hidden = output
                else:
                    # Most models return (logits, loss)
                    logits, loss = output
                
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Check for NaN loss (edge case: all labels are -100)
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx} in epoch {epoch+1}")
                print(f"  Valid targets: {(labels != -100).sum().item()}")
                print(f"  Total targets: {labels.numel()}")
                print(f"  Skipping this batch")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'],
                                     epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                    # Models that accept targets parameter
                    output = self.model(input_ids, targets=labels)
                    if len(output) == 3:
                        # RNN-based models return (logits, loss, hidden)
                        logits, loss, hidden = output
                    else:
                        # Transformer models return (logits, loss)
                        logits, loss = output
                    
                    if loss is None:
                        # If model doesn't compute loss, compute it manually
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    # Models that don't accept targets parameter
                    output = self.model(input_ids)
                    if len(output) == 3:
                        # RNN-based models return (logits, loss, hidden)
                        logits, loss, hidden = output
                    else:
                        # Transformer models return (logits, loss)
                        logits, loss = output
                    
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'args': self.args
        }
        
        # Create timestamp-based directory for this experiment
        timestamp = self.logger.timestamp
        exp_dir = f"checkpoints/{self.args.model}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{exp_dir}/epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = f"{exp_dir}/best.pt"
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved to {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training of {self.args.model} model")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.args.epochs}")
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Learning rate: {self.args.learning_rate}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch-level metrics to TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            
            # Calculate and log perplexity (exponential of loss)
            # Check for NaN losses before calculating perplexity
            if torch.isnan(torch.tensor(train_loss)):
                print(f"Warning: NaN train loss detected at epoch {epoch+1}")
                train_perplexity = torch.tensor(float('inf'))
            else:
                train_perplexity = torch.exp(torch.tensor(train_loss))
            
            if torch.isnan(torch.tensor(val_loss)):
                print(f"Warning: NaN validation loss detected at epoch {epoch+1}")
                val_perplexity = torch.tensor(float('inf'))
            else:
                val_perplexity = torch.exp(torch.tensor(val_loss))
            
            self.writer.add_scalar('Epoch/Train_Perplexity', train_perplexity, epoch)
            self.writer.add_scalar('Epoch/Val_Perplexity', val_perplexity, epoch)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Best Val Loss: {self.best_val_loss:.4f}")
            print(f"   Learning Rate: {current_lr:.2e}")
        
        # Save final log
        self.logger.save_log()
        summary = self.logger.get_summary()
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Model config saved to: {self.logger.log_file}")
        print(f"   TensorBoard logs saved to: logs/runs/{self.args.model}_{self.logger.timestamp}/")
        print(f"   Checkpoints saved to: checkpoints/{self.args.model}_{self.logger.timestamp}/")
        print(f"\n📊 TensorBoard Metrics Available:")
        print(f"   - Train/Loss: Batch-level training losses")
        print(f"   - Val/Loss: Epoch-level validation losses")
        print(f"   - Epoch/Train_Loss: Epoch-level training losses")
        print(f"   - Epoch/Val_Loss: Epoch-level validation losses")
        print(f"   - Epoch/Train_Perplexity: Training perplexity")
        print(f"   - Epoch/Val_Perplexity: Validation perplexity")
        
        # Close tensorboard writer
        self.writer.close()


def create_model(model_name, vocab_size, args):
    """Create a model based on the model name and arguments."""
    if model_name == "gpt_encoder":
        config = GPTEncoderConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            p_drop=args.dropout
        )
        return GPTEncoderLayerLM(config)
    
    elif model_name == "mini_gpt":
        config = MiniGPTConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            p_drop=args.dropout
        )
        return GPTDecoderLM(config)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a language model on WikiText-2")
    
    # Model selection
    parser.add_argument("--model", type=str, required=True,
                       choices=["gpt_encoder", "mini_gpt"],
                       help="Model to train")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=256,
                       help="Model dimension")
    parser.add_argument("--n_layer", type=int, default=4,
                       help="Number of layers")
    parser.add_argument("--n_head", type=int, default=8,
                       help="Number of attention heads (for transformer models)")
    parser.add_argument("--d_ff", type=int, default=1024,
                       help="Feed-forward dimension (for transformer models)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Data parameters
    parser.add_argument("--seq_len", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for datasets")
    
    # System parameters
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loader workers")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Logging interval")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Setup")
    print(f"   Model: {args.model}")
    print(f"   Device: {device}")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Batch size: {args.batch_size}")
    
    # Load dataset
    print(f"\n📚 Loading WikiText-2 dataset...")
    data = build_wikitext2(
        tokenizer_name=args.tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir
    )
    
    print(f"   Vocab size: {data['vocab_size']}")
    print(f"   Train batches: {len(data['train_loader'])}")
    print(f"   Val batches: {len(data['val_loader'])}")
    
    # Create model
    print(f"\n🏗️  Creating {args.model} model...")
    model = create_model(args.model, data['vocab_size'], args)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        device=device,
        args=args
    )
    
    # Update logger with actual vocab size
    trainer.update_logger_config(data['vocab_size'])
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
