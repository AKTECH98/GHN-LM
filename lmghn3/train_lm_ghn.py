#!/usr/bin/env python3
"""
Train a Graph HyperNetwork (GHN-3) for Language Models

This script trains a GHN to predict parameters for various language model architectures
(RNN, LSTM, GRU, GPT Encoder, Mini GPT) using the WikiText-2 dataset.

Example usage:
    # Train GHN-3 on language models with reasonable dataset
    python train_lm_ghn.py --name ghn3-lm --epochs 50 --batch_size 4 --meta_batch_size 8 \
        --lr 1e-4 --wd 1e-2 --hid 64 --layers 3 --heads 8 --amp

    # Train with full dataset (3M+ configurations)
    python train_lm_ghn.py --name ghn3-lm-full --use_all_configs --epochs 100 \
        --batch_size 2 --meta_batch_size 4 --lr 5e-5 --wd 1e-2 --hid 128 --layers 4

    # Train with different GHN architecture options
    python train_lm_ghn.py --name ghn3-lm-advanced --hypernet gnn --decoder mlp \
        --weight_norm --ve --layernorm --epochs 30 --hid 96 --layers 4 --heads 12

    # Train GHN-2 instead of GHN-3
    python train_lm_ghn.py --name ghn2-lm --is_ghn2 --epochs 40 --hid 64 --layers 3
"""

import argparse
import time
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import json
from datetime import datetime

# Add paths
sys.path.append('lmghn3')

from lmghn3.Dataloader import (
    create_reasonable_model_dataloader,
    create_full_model_dataloader,
    build_wikitext2
)
from lmghn3.CustomGHN3.nn import GHN3
from lmghn3.CustomGHN3.trainer import Trainer
from lmghn3.CustomGHN3.utils import log
from lmghn3.CustomGHN3.graph import Graph, GraphBatch
from simple_logger import SimpleLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GHN-3 for Language Models')
    
    # Model configuration
    parser.add_argument('--name', type=str, default='ghn3-lm', help='experiment name')
    parser.add_argument('--hid', type=int, default=64, help='hidden dimension of GHN')
    parser.add_argument('--layers', type=int, default=3, help='number of GHN layers')
    parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--max_shape', type=int, nargs=4, default=[64, 64, 11, 11], 
                       help='max shape for GHN (d_model, d_model, seq_len, seq_len)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for language models')
    parser.add_argument('--meta_batch_size', type=int, default=8, help='number of models per meta-batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine-warmup', 
                       choices=['cosine', 'cosine-warmup', 'step'], help='learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')
    
    # Dataset configuration
    parser.add_argument('--use_all_configs', action='store_true', 
                       help='use full dataset (3M+ configs) instead of reasonable (~17K)')
    parser.add_argument('--vocab_size', type=int, default=50257, help='vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--data_workers', type=int, default=2, help='number of data loading workers')
    
    # Training options
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='logging interval')
    parser.add_argument('--save_interval', type=int, default=100, help='checkpoint saving interval')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='logs/ghn_training', help='directory to save checkpoints')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to resume from')
    
    # GHN specific
    parser.add_argument('--predparam_wd', type=float, default=3e-5, help='predicted parameter weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')
    parser.add_argument('--compile', type=str, default=None, help='pytorch compilation mode')
    
    # GHN architecture parameters
    parser.add_argument('--hypernet', type=str, default='gatedgnn', 
                       choices=['gatedgnn', 'gnn'], help='hypernetwork type')
    parser.add_argument('--decoder', type=str, default='conv', 
                       choices=['conv', 'mlp'], help='decoder type')
    parser.add_argument('--weight_norm', action='store_true', help='use weight normalization')
    parser.add_argument('--ve', action='store_true', help='use virtual edges')
    parser.add_argument('--layernorm', action='store_true', help='use layer normalization')
    parser.add_argument('--is_ghn2', action='store_true', help='use GHN-2 architecture instead of GHN-3')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained embeddings')
    
    return parser.parse_args()


class LanguageModelTrainer:
    """Trainer for GHN on language models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
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
                'ghn_type': 'GHN-2' if self.args.is_ghn2 else 'GHN-3',
                'hypernet': self.args.hypernet,
                'decoder': self.args.decoder,
                'weight_norm': self.args.weight_norm,
                'virtual_edges': self.args.ve,
                'layer_norm': self.args.layernorm,
                'pretrained': self.args.pretrained,
                'use_all_configs': self.args.use_all_configs,
                'vocab_size': self.args.vocab_size,
                'max_seq_len': self.args.max_seq_len,
            },
            args=self.args
        )
        
        log(f'TensorBoard logs: {tb_dir}')
        log(f'Metadata logs: logs/training_logs/')
        
    def _setup_data(self):
        """Setup data loaders."""
        log('Setting up data loaders...')
        
        # Create model dataloader
        if self.args.use_all_configs:
            log('Using full dataset (3M+ configurations)...')
            self.model_dataloader, self.model_configs = create_full_model_dataloader(
                vocab_size=self.args.vocab_size,
                max_seq_len=self.args.max_seq_len,
                batch_size=self.args.meta_batch_size,
                num_workers=0,  # No multiprocessing for model creation
                device=self.device,
                seed=self.args.seed
            )
        else:
            log('Using reasonable dataset (~17K configurations)...')
            self.model_dataloader, self.model_configs = create_reasonable_model_dataloader(
                vocab_size=self.args.vocab_size,
                max_seq_len=self.args.max_seq_len,
                batch_size=self.args.meta_batch_size,
                num_workers=0,
                device=self.device,
                seed=self.args.seed
            )
        
        # Create WikiText-2 dataloader
        log('Loading WikiText-2 dataset...')
        self.data_loaders = build_wikitext2(
            tokenizer_name="gpt2",
            seq_len=self.args.max_seq_len,
            batch_size=self.args.batch_size,
            num_workers=self.args.data_workers
        )
        
        self.train_loader = self.data_loaders['train_loader']
        self.val_loader = self.data_loaders['val_loader']
        
        log(f'Model dataset: {len(self.model_configs)} configurations')
        log(f'Training batches: {len(self.train_loader)}')
        log(f'Validation batches: {len(self.val_loader)}')
        
    def _setup_ghn(self):
        """Setup GHN model."""
        log('Setting up GHN model...')
        
        # GHN configuration
        ghn_config = {
            'max_shape': tuple(self.args.max_shape),
            'num_classes': self.args.vocab_size,
            'hypernet': self.args.hypernet,
            'decoder': self.args.decoder,
            'weight_norm': self.args.weight_norm,
            've': self.args.ve,
            'layernorm': self.args.layernorm,
            'hid': self.args.hid,
            'layers': self.args.layers,
            'heads': self.args.heads,
            'is_ghn2': self.args.is_ghn2,
            'pretrained': self.args.pretrained,
        }
        
        self.ghn = GHN3(**ghn_config, debug_level=0)
        self.ghn.to(self.device)
        
        total_params = sum(p.numel() for p in self.ghn.parameters())
        log(f'GHN created with {total_params:,} parameters')
        
    def _setup_trainer(self):
        """Setup trainer."""
        log('Setting up trainer...')
        
        # Optimizer arguments
        opt_args = {
            'lr': self.args.lr,
            'weight_decay': self.args.wd
        }
        if self.args.opt == 'sgd':
            opt_args['momentum'] = 0.9
        
        # Scheduler arguments
        scheduler_args = {}
        if self.args.scheduler == 'step':
            scheduler_args = {'milestones': [30, 40], 'gamma': 0.1}
        
        self.trainer = Trainer(
            model=self.ghn,
            opt=self.args.opt,
            opt_args=opt_args,
            scheduler=self.args.scheduler,
            scheduler_args=scheduler_args,
            n_batches=len(self.train_loader),
            grad_clip=self.args.grad_clip,
            device=self.device,
            log_interval=self.args.log_interval,
            amp=self.args.amp,
            amp_min_scale=1024,
            amp_growth_interval=100,
            predparam_wd=self.args.predparam_wd,
            label_smoothing=self.args.label_smoothing,
            save_dir=self.args.save_dir,
            ckpt=self.args.ckpt,
            epochs=self.args.epochs,
            verbose=True,
            compile_mode=self.args.compile
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        log(f'\nEpoch {epoch+1}/{self.args.epochs}, LR: {self.trainer.get_lr():.2e}')
        
        self.trainer.reset_metrics(epoch)
        self.ghn.train()
        
        # Create iterator for model batches
        model_iter = iter(self.model_dataloader)
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids']
            targets = batch['labels']
            try:
                # Get next batch of models
                try:
                    models, metadatas = next(model_iter)
                except StopIteration:
                    # Restart model iterator
                    model_iter = iter(self.model_dataloader)
                    models, metadatas = next(model_iter)
                
                # Move data to device
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # For language modeling, targets should be flattened to match model output
                # Model output: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
                # Targets: [batch_size, seq_len] -> [batch_size * seq_len]
                targets = targets.view(-1)  # Flatten targets
                
                # Create graphs for GHN
                from lmghn3.CustomGHN3.graph_adapter import create_language_model_graphs
                graphs = create_language_model_graphs(models, metadatas)
                
                # Update trainer with language model specific handling
                self._update_language_model_step(input_ids, targets, graphs, step, epoch)
                
                # Save checkpoint
                if self.args.save_dir and (step + 1) % self.args.save_interval == 0:
                    self.trainer.save(epoch, step, {'args': self.args})
                    
            except Exception as e:
                log(f'Error at step {step}: {e}')
                continue
                
        # Step scheduler
        self.trainer.scheduler_step()
        
        # Epoch-level logging
        avg_loss = self.trainer.metrics['loss'].avg
        log(f'Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}')
        
        # TensorBoard epoch-level logging
        self.writer.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Learning_Rate', self.trainer.get_lr(), epoch)
        
    def _update_language_model_step(self, input_ids, targets, graphs, step, epoch):
        """Custom training step for language models with consistent input/output shapes."""
        try:
            # Get predicted models from GHN
            predicted_models = self.ghn(graphs.nets, graphs.to_device(self.device),
                                      bn_track_running_stats=True, keep_grads=True, reduce_graph=True)
            
            # Standardize input/output shapes for all models
            batch_size, seq_len = input_ids.shape
            vocab_size = self.args.vocab_size
            
            # Process each model with consistent shapes
            total_loss = 0
            num_models = 0
            
            for model in predicted_models:
                try:
                    # Forward pass - all models should output [batch_size, seq_len, vocab_size]
                    output = model(input_ids)
                    
                    # Handle tuple output (logits, hidden_states)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    
                    # Ensure consistent output shape: [batch_size, seq_len, vocab_size]
                    if logits.dim() == 3 and logits.shape == (batch_size, seq_len, vocab_size):
                        # Reshape to [batch_size * seq_len, vocab_size] for loss calculation
                        logits = logits.view(-1, vocab_size)
                    else:
                        log(f"Warning: Unexpected output shape {logits.shape}, expected ({batch_size}, {seq_len}, {vocab_size})")
                        continue
                    
                    # Calculate loss with consistent shapes
                    # logits: [batch_size * seq_len, vocab_size]
                    # targets: [batch_size * seq_len]
                    loss = self.trainer.criterion(logits, targets)
                    total_loss += loss
                    num_models += 1
                    
                except Exception as e:
                    log(f"Error processing model: {e}")
                    continue
            
            if num_models > 0:
                # Average loss across models
                avg_loss = total_loss / num_models
                
                # Backward pass
                self.trainer._optimizer.zero_grad()
                avg_loss.backward()
                
                # Gradient clipping
                if self.trainer.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.ghn.parameters(), self.trainer.grad_clip)
                
                # Optimizer step
                self.trainer._optimizer.step()
                
                # Update metrics
                self.trainer.metrics['loss'].update(avg_loss.item(), len(targets))
                
                # Log progress
                if step % self.trainer.log_interval == 0:
                    log(f'Step {step}: Loss = {avg_loss.item():.4f} (processed {num_models} models)')
                    
                    # TensorBoard logging
                    global_step = epoch * len(self.train_loader) + step
                    self.writer.add_scalar('Train/Loss', avg_loss.item(), global_step)
                    self.writer.add_scalar('Train/Learning_Rate', self.trainer.get_lr(), global_step)
                    self.writer.add_scalar('Train/Models_Processed', num_models, global_step)
            else:
                log(f'Warning: No models processed successfully at step {step}')
            
        except Exception as e:
            log(f'Error in training step: {e}')
            import traceback
            traceback.print_exc()
        
        
    def train(self):
        """Main training loop."""
        log(f'Starting training for {self.args.epochs} epochs...')
        log(f'Device: {self.device}')
        log(f'Model dataset size: {len(self.model_configs)}')
        
        start_time = time.time()
        
        for epoch in range(self.trainer.start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            
            # Log epoch summary
            metrics = {name: meter.avg for name, meter in self.trainer.metrics.items()}
            log(f'Epoch {epoch+1} completed - Loss: {metrics["loss"]:.4f}, '
                f'Top1: {metrics["top1"]:.2f}%, Top5: {metrics["top5"]:.2f}%')
        
        total_time = time.time() - start_time
        log(f'Training completed in {total_time/3600:.2f} hours')
        
        # Final TensorBoard logging
        self.writer.add_scalar('Training/Total_Time_Hours', total_time/3600, 0)
        self.writer.add_scalar('Training/Total_Epochs', self.args.epochs, 0)
        self.writer.add_scalar('Training/Model_Dataset_Size', len(self.model_configs), 0)
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save final checkpoint
        if self.args.save_dir:
            final_path = os.path.join(self.args.save_dir, 'final_checkpoint.pt')
            torch.save({
                'state_dict': self.ghn.state_dict(),
                'args': self.args,
                'model_configs': len(self.model_configs)
            }, final_path)
            log(f'Final checkpoint saved to {final_path}')
            
        # Log final metadata
        final_metrics = {name: meter.avg for name, meter in self.trainer.metrics.items()}
        self.metadata_logger.log_final({
            'final_loss': final_metrics.get('loss', 0.0),
            'total_time_hours': total_time/3600,
            'total_epochs': self.args.epochs,
            'model_dataset_size': len(self.model_configs)
        })


def main():
    """Main function."""
    args = parse_args()
    
    # Print configuration
    log('=' * 60)
    log('GHN-3 Language Model Training')
    log('=' * 60)
    log(f'Experiment: {args.name}')
    log(f'GHN Config: hid={args.hid}, layers={args.layers}, heads={args.heads}')
    log(f'Training: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}')
    log(f'Dataset: {"Full" if args.use_all_configs else "Reasonable"} ({len(args.model_configs) if hasattr(args, "model_configs") else "?"} configs)')
    log('=' * 60)
    
    # Create trainer and start training
    trainer = LanguageModelTrainer(args)
    trainer.train()
    
    log('Training completed successfully!')


if __name__ == '__main__':
    main()
