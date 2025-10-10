#!/usr/bin/env python3
"""
Train a Graph HyperNetwork (GHN-3) for Language Models

This script trains a GHN-3 to predict parameters for various language model architectures
(RNN, LSTM, GRU, GPT Encoder, Mini GPT) using the WikiText-2 dataset.

Example usage:
    # Train GHN-3 on language models with reasonable dataset
    python train_ghn_lm.py --name ghn3-lm --epochs 50 --batch_size 4 --meta_batch_size 8 \
        --lr 1e-4 --wd 1e-2 --hid 64 --layers 3 --heads 8 --amp

    # Train with full dataset (3M+ configurations)
    python train_ghn_lm.py --name ghn3-lm-full --use_all_configs --epochs 100 \
        --batch_size 2 --meta_batch_size 4 --lr 5e-5 --wd 1e-2 --hid 128 --layers 4

    # Train with different GHN architecture options
    python train_ghn_lm.py --name ghn3-lm-advanced --hypernet gnn --decoder mlp \
        --weight_norm --ve --layernorm --epochs 30 --hid 96 --layers 4 --heads 12
"""

import argparse
import time
import os
import sys
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add paths
sys.path.append('.')

from lmghn3.Dataloader import (
    create_reasonable_model_dataloader,
    create_full_model_dataloader,
    build_wikitext2
)
from lmghn3.CustomGHN3.nn import GHN3
from lmghn3.CustomGHN3.trainer import Trainer
from lmghn3.CustomGHN3.utils import log
from lmghn3.CustomGHN3.language_model_loader import LanguageModelLoader
from simple_logger import SimpleLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GHN-3 Language Model Training')
    
    # Experiment settings
    parser.add_argument('--name', type=str, default='ghn3-lm', help='experiment name')
    parser.add_argument('--save', type=str, default='./checkpoints', help='save directory')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for language models')
    parser.add_argument('--meta_batch_size', type=int, default=8, help='number of models per meta-batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')
    
    # GHN architecture settings
    parser.add_argument('--hid', type=int, default=64, help='GHN hidden dimension')
    parser.add_argument('--layers', type=int, default=3, help='number of GHN layers')
    parser.add_argument('--heads', type=int, default=8, help='number of attention heads in GHN')
    parser.add_argument('--hypernet', type=str, default='gatedgnn', choices=['gatedgnn', 'gnn'], help='hypernetwork type')
    parser.add_argument('--decoder', type=str, default='conv', choices=['conv', 'mlp'], help='decoder type')
    parser.add_argument('--weight_norm', action='store_true', help='use weight normalization')
    parser.add_argument('--ve', action='store_true', help='use virtual edges')
    parser.add_argument('--layernorm', action='store_true', help='use layer normalization')
    
    # Language model settings
    parser.add_argument('--vocab_size', type=int, default=50257, help='vocabulary size (will be overridden by actual tokenizer vocab size)')
    parser.add_argument('--max_seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--use_all_configs', action='store_true', help='use full dataset (3M+ configs) or reasonable (~17K)')
    
    # Data settings
    parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--cache_dir', type=str, default=None, help='cache directory for datasets')
    
    # Logging settings
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--verbose', action='store_true', help='verbose logging')
    
    return parser.parse_args()


class LanguageModelTrainer:
    """Trainer for GHN on language models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Initialize best metrics tracking
        self.best_val_loss = float('inf')
        self.best_perplexity = float('inf')
        
        # Create save directory if specified
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._setup_data()
        self._setup_ghn()
        self._setup_trainer()
        
    def _setup_logging(self):
        """Setup TensorBoard and metadata logging."""
        # Create TensorBoard writer
        tb_dir = f"runs/{self.args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(tb_dir)
        
        # Create metadata logger
        self.metadata_logger = SimpleLogger(
            model_name=f"ghn3_{self.args.name}",
            config={
                'experiment_name': self.args.name,
                'ghn_type': 'GHN-3',
                'hypernet': self.args.hypernet,
                'decoder': self.args.decoder,
                'weight_norm': self.args.weight_norm,
                'virtual_edges': self.args.ve,
                'layer_norm': self.args.layernorm,
                'vocab_size': self.args.vocab_size,
                'max_seq_len': self.args.max_seq_len,
                'use_all_configs': self.args.use_all_configs,
            },
            args=self.args
        )
        
        log(f'TensorBoard logs: {tb_dir}')
        log(f'Metadata logs: logs/training_logs/')
        
    def _setup_data(self):
        """Setup data loaders."""
        log('Setting up data loaders...')
        
        # Setup WikiText-2 data
        self.data_config = build_wikitext2(
            tokenizer_name="gpt2",
            seq_len=self.args.max_seq_len,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            cache_dir=self.args.cache_dir
        )
        
        # Get actual vocabulary size from the tokenizer
        actual_vocab_size = self.data_config['vocab_size']
        log(f'WikiText-2 vocab size: {actual_vocab_size}')
        
        # Use the actual vocabulary size from the tokenizer instead of the argument
        if self.args.vocab_size != actual_vocab_size:
            log(f'Overriding vocab_size argument ({self.args.vocab_size}) with actual tokenizer vocab_size ({actual_vocab_size})')
            self.args.vocab_size = actual_vocab_size
        
        # Setup model architectures
        if self.args.use_all_configs:
            log('Using full dataset (3M+ configurations)')
            _, self.model_configs = create_full_model_dataloader(
                vocab_size=self.args.vocab_size,
                max_seq_len=self.args.max_seq_len,
                batch_size=self.args.meta_batch_size,
                num_workers=self.args.num_workers,
                device=self.device
            )
        else:
            log('Using reasonable dataset (~17K configurations)')
            _, self.model_configs = create_reasonable_model_dataloader(
                vocab_size=self.args.vocab_size,
                max_seq_len=self.args.max_seq_len,
                batch_size=self.args.meta_batch_size,
                num_workers=self.args.num_workers,
                device=self.device
            )
        
        log(f'Model configurations: {len(self.model_configs)}')
        log(f'Meta batch size: {self.args.meta_batch_size}')
        
    def _setup_ghn(self):
        """Setup GHN-3 model."""
        log('Setting up GHN-3 model...')
        
        # GHN configuration
        # For language models, we need to estimate max_shape based on typical LM sizes
        # Analyze all model configs to find the maximum hidden size
        max_hidden = 0
        for cfg in self.model_configs:
            # Check different possible hidden size keys
            hidden_size = max(
                cfg.get('d_model', 0),
                cfg.get('hidden_size', 0), 
                cfg.get('n_embd', 0),
                cfg.get('d_hidden', 0),
                cfg.get('hidden_dim', 0)
            )
            max_hidden = max(max_hidden, hidden_size)
        
        # Ensure we have a reasonable minimum
        max_hidden = max(max_hidden, 128)
        max_vocab = self.args.vocab_size
        
        # For language models, max_shape should be (vocab_size, 4*hidden_size, 1, 1)
        # This accounts for the largest possible parameter tensors in language models
        max_shape = (4*max_hidden, 4 * max_hidden, 1, 1)  # (vocab_size, 4*hidden_size, 1, 1)
        
        log(f'Max hidden size found: {max_hidden}')
        log(f'Vocab size: {max_vocab}')
        log(f'Calculated max_shape: {max_shape}')
        
        ghn_config = {
            'max_shape': max_shape,
            'num_classes': max_vocab,  # For language models, this is vocab size
            'hypernet': self.args.hypernet,
            'decoder': self.args.decoder,
            'weight_norm': self.args.weight_norm,
            've': self.args.ve,
            'layernorm': self.args.layernorm,
            'hid': self.args.hid,
            'layers': self.args.layers,
            'heads': self.args.heads,
            'is_ghn2': False,
        }
        
        self.ghn = GHN3(**ghn_config, debug_level=1)
        
        log(f'GHN-3 parameters: {sum(p.numel() for p in self.ghn.parameters()):,}')
        log(f'GHN-3 config: {ghn_config}')
        
    def _setup_trainer(self):
        """Setup trainer."""
        log('Setting up trainer...')
        
        self.trainer = Trainer(
            self.ghn,
            opt='adamw',
            opt_args={'lr': self.args.lr, 'weight_decay': self.args.wd},
            scheduler='cosine-warmup',
            scheduler_args={'warmup_steps': 5, 'init_lr': 1e-5},
            n_batches=len(self.data_config['train_loader']),
            grad_clip=self.args.grad_clip,
            device=self.device,
            log_interval=self.args.log_interval,
            amp=self.args.amp,
            amp_min_scale=1024,
            amp_growth_interval=100,
            predparam_wd=3e-5,  # Parameter regularization
            label_smoothing=0.0,  # No label smoothing for language models
            save_dir=self.args.save,
            ckpt=self.args.ckpt,
            epochs=self.args.epochs,
            verbose=self.args.verbose
        )
        
    def train(self):
        """Main training loop."""
        log(f'Starting GHN-3 training for language models')
        log(f'Device: {self.device}')
        log(f'Epochs: {self.args.epochs}')
        log(f'Batch size: {self.args.batch_size}')
        log(f'Meta batch size: {self.args.meta_batch_size}')
        log(f'Learning rate: {self.args.lr}')
        
        # Create iterators
        train_loader = self.data_config['train_loader']
        
        # Setup graph loader using the new direct loading approach
        graphs_loader = LanguageModelLoader.loader(
            model_configs=self.model_configs,
            meta_batch_size=self.args.meta_batch_size,
            dense=True,
            device=self.device,
            num_workers=0  # Use 0 for simplicity
        )
        
        # Setup graph loader iterator
        graphs_iter = iter(graphs_loader)
        
        for epoch in range(self.trainer.start_epoch, self.args.epochs):
            log(f'\nEpoch {epoch+1}/{self.args.epochs}, LR: {self.trainer.get_lr():.2e}')
            
            self.trainer.reset_metrics(epoch)
            
            for step, batch in enumerate(train_loader, start=self.trainer.start_step):
                if step >= len(train_loader):
                    break
                
                # Extract input_ids and labels from batch
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Get next batch of graphs
                try:
                    graphs = next(graphs_iter)
                except StopIteration:
                    # Restart graph loader
                    graphs_iter = iter(graphs_loader)
                    graphs = next(graphs_iter)
                
                # Train GHN-3
                try:
                    # Use the language model update method
                    metrics = self.trainer.update_lm(input_ids, labels, graphs=graphs)
                    
                    if metrics is None:
                        log(f'Skipped batch at step {step} due to NaN loss')
                        continue
                        
                except Exception as e:
                    log(f'Error in training step {step}: {e}')
                    continue
                
                # Log progress
                self.trainer.log(step)
                
                # Add batch-level TensorBoard logging
                if step % self.args.log_interval == 0:
                    global_step = epoch * len(self.data_config['train_loader']) + step
                    if metrics and 'loss' in metrics:
                        # Extract scalar value from AvgrageMeter
                        loss_value = metrics['loss'].avg if hasattr(metrics['loss'], 'avg') else metrics['loss']
                        self.writer.add_scalar('Train/Loss', loss_value, global_step)
                    if metrics and 'top1' in metrics:
                        # top1 stores perplexity for language models
                        perplexity_value = metrics['top1'].avg if hasattr(metrics['top1'], 'avg') else metrics['top1']
                        self.writer.add_scalar('Train/Perplexity', perplexity_value, global_step)
                    
                    # Log learning rate
                    if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                        current_lr = self.trainer.optimizer.param_groups[0]['lr']
                        self.writer.add_scalar('Train/LR', current_lr, global_step)
                
                # Note: Checkpoint saving moved to epoch-level for better organization
            
            # Scheduler step
            self.trainer.scheduler_step()
            
            # Run validation
            val_loss, val_perplexity = self.validate(epoch)
            
            # Log epoch summary
            metrics = {metric: value.avg for metric, value in self.trainer.metrics.items()}
            train_loss = metrics.get('loss', 0.0)
            train_perplexity = metrics.get('top1', 0.0)  # top1 stores perplexity for language models
            
            # Calculate epoch-level perplexity with NaN checks
            if torch.isnan(torch.tensor(train_loss)):
                log(f"Warning: NaN train loss detected at epoch {epoch+1}")
                train_perplexity = torch.tensor(float('inf'))
            else:
                train_perplexity = torch.exp(torch.tensor(train_loss))
            
            if torch.isnan(torch.tensor(val_loss)):
                log(f"Warning: NaN validation loss detected at epoch {epoch+1}")
                val_perplexity = torch.tensor(float('inf'))
            else:
                val_perplexity = torch.exp(torch.tensor(val_loss))
            
            # Log epoch-level metrics to TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Perplexity', train_perplexity, epoch)
            self.writer.add_scalar('Epoch/Val_Perplexity', val_perplexity, epoch)
            
            # Update best metrics
            is_best_val = val_loss < self.best_val_loss
            is_best_perplexity = val_perplexity < self.best_perplexity
            if is_best_val:
                self.best_val_loss = val_loss
            if is_best_perplexity:
                self.best_perplexity = val_perplexity
            
            # Save checkpoint
            if self.args.save:
                self.save_checkpoint(epoch, val_loss, is_best_val)
            
            # Print detailed epoch summary
            log(f'\n📊 Epoch {epoch+1} Summary:')
            log(f'   Train Loss: {train_loss:.4f}')
            log(f'   Val Loss: {val_loss:.4f}')
            log(f'   Train Perplexity: {train_perplexity:.4f}')
            log(f'   Val Perplexity: {val_perplexity:.4f}')
            log(f'   Best Val Loss: {self.best_val_loss:.4f}')
            log(f'   Best Perplexity: {self.best_perplexity:.4f}')
            
            # Log current learning rate
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                log(f'   Learning Rate: {current_lr:.2e}')
                self.writer.add_scalar('Epoch/LR', current_lr, epoch)
        
        # Save final log
        self.metadata_logger.save_log()
        
        # Print final training summary
        log(f'\n🎉 Training completed at {time.strftime("%Y%m%d-%H%M%S")}!')
        log(f'   Best validation loss: {self.best_val_loss:.4f}')
        log(f'   Best perplexity: {self.best_perplexity:.4f}')
        log(f'   Model config saved to: {self.metadata_logger.log_file}')
        log(f'   TensorBoard logs saved to: runs/{self.args.name}_*/')
        if self.args.save:
            timestamp = self.metadata_logger.timestamp
            checkpoint_dir = f"{self.args.save}/ghn3_{self.args.name}_{timestamp}"
            log(f'   Checkpoints saved to: {checkpoint_dir}/')
        log(f'\n📊 TensorBoard Metrics Available:')
        log(f'   - Train/Loss: Training loss per batch')
        log(f'   - Train/Perplexity: Training perplexity per batch')
        log(f'   - Train/LR: Learning rate per batch')
        log(f'   - Val/Loss: Validation loss per epoch')
        log(f'   - Val/Perplexity: Validation perplexity per epoch')
        log(f'   - Epoch/Train_Loss: Average training loss per epoch')
        log(f'   - Epoch/Val_Loss: Average validation loss per epoch')
        log(f'   - Epoch/Train_Perplexity: Average training perplexity per epoch')
        log(f'   - Epoch/Val_Perplexity: Average validation perplexity per epoch')
        log(f'   - Epoch/LR: Learning rate per epoch')
        log(f'\n🚀 To view TensorBoard: tensorboard --logdir runs/')
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save GHN model checkpoint with comprehensive metadata."""
        checkpoint = {
            'epoch': epoch,
            'ghn_state_dict': self.ghn.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict() if self.trainer.optimizer else None,
            'scheduler_state_dict': self.trainer.scheduler.state_dict() if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'best_perplexity': self.best_perplexity,
            'args': self.args,
            'ghn_config': {
                'max_shape': self.ghn.max_shape,
                'num_classes': self.ghn.num_classes,
                'hid': self.args.hid,
                'layers': self.args.layers,
                'heads': self.args.heads,
                'hypernet': self.args.hypernet,
                'decoder': self.args.decoder,
                'weight_norm': self.args.weight_norm,
                'virtual_edges': self.args.ve,
                'layer_norm': self.args.layernorm,
            },
            'model_configs_count': len(self.model_configs),
            'vocab_size': self.args.vocab_size,
            'max_seq_len': self.args.max_seq_len,
        }
        
        # Create timestamp-based directory for this experiment
        timestamp = self.metadata_logger.timestamp
        exp_dir = f"{self.args.save}/ghn3_{self.args.name}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{exp_dir}/epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            log(f'💾 Checkpoint saved to {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = f"{exp_dir}/best.pt"
            torch.save(checkpoint, best_path)
            log(f'✅ New best model saved to {best_path}')
    
    def validate(self, epoch):
        """Validate the GHN model on validation data."""
        self.ghn.eval()
        total_loss = 0
        total_perplexity = 0
        num_batches = 0
        
        with torch.no_grad():
            # Create validation data iterator
            val_data_iter = iter(self.data_config['val_loader'])
            
            # Create validation graphs iterator
            val_graphs_loader = LanguageModelLoader.loader(
                model_configs=self.model_configs,
                meta_batch_size=self.args.meta_batch_size,
                dense=True,
                device=self.device,
                num_workers=0
            )
            val_graphs_iter = iter(val_graphs_loader)
            
            for step in range(len(self.data_config['val_loader'])):
                try:
                    # Get next batch of validation data
                    batch = next(val_data_iter)
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Get next batch of graphs
                    try:
                        graphs = next(val_graphs_iter)
                    except StopIteration:
                        val_graphs_iter = iter(val_graphs_loader)
                        graphs = next(val_graphs_iter)
                    
                    # Validate GHN-3
                    metrics = self.trainer.update_lm(input_ids, labels, graphs=graphs)
                    
                    if metrics and 'loss' in metrics:
                        total_loss += metrics['loss']
                        if 'perplexity' in metrics:
                            total_perplexity += metrics['perplexity']
                        num_batches += 1
                        
                except Exception as e:
                    log(f'Error in validation step {step}: {e}')
                    continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_perplexity = total_perplexity / num_batches if total_perplexity > 0 else float('inf')
            
            # Log validation metrics to TensorBoard
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.writer.add_scalar('Val/Perplexity', avg_perplexity, epoch)
            
            return avg_loss, avg_perplexity
        else:
            log(f'Warning: No valid validation batches processed')
            return float('inf'), float('inf')


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create trainer and start training
    trainer = LanguageModelTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
