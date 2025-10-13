# Copyright (c) 2023. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains a Graph HyperNetwork (GHN-3) for Language Models. DistributedDataParallel (DDP) training is
used if `torchrun` is used as shown below.
This script uses curated language model architectures and WikiText-2 dataset for training.

Example:

    # To train GHN-3 for Language Models on single GPU, automatic mixed precision:
    python train_ghn_ddp.py --vocab_size 50257 --seq_len 256 --ln \
    -e 20 --opt adamw --lr 4e-4 --wd 1e-2 -b 8 --amp -m 8 --name ghn3lm --hid 256 --scheduler cosine-warmup

    # 4 GPUs (DDP), automatic mixed precision:
    export OMP_NUM_THREADS=8
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ghn_ddp.py --vocab_size 50257 --seq_len 256 --ln \
    -e 20 --opt adamw --lr 4e-4 --wd 1e-2 -b 8 --amp -m 8 --name ghn3lm --hid 256 --scheduler cosine-warmup

    # Use eval_ghn.py to evaluate the trained GHN-3 model on language modeling tasks.

"""


import argparse
import time
import sys
import os
import warnings
import json
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import uuid
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path to import language model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import warning suppression
from suppress_warnings import suppress_training_warnings
suppress_training_warnings()

from ppuda.config import init_config
from CustomGHN3 import GHN3, log, Trainer, setup_ddp, clean_ddp
from lmghn3.language_models.lm_arch_loader import build_ghn_variants_dataloader
from lmghn3.language_models.wikitext2_loader import build_wikitext2

log = partial(log, flush=True)


def create_experiment_metadata(args, config, run_id, start_time):
    """Create comprehensive experiment metadata for tracking and comparison."""
    metadata = {
        'experiment_info': {
            'run_id': run_id,
            'start_time': start_time.isoformat(),
            'experiment_name': args.name,
            'description': f'GHN-3 training for language models - {args.name}'
        },
        'model_config': {
            'ghn_type': 'GHN-3' if not args.ghn2 else 'GHN-2',
            'hidden_dim': args.hid,
            'num_layers': args.layers,
            'num_heads': args.heads,
            'max_shape': args.max_shape,
            'num_classes': args.vocab_size,
            'compile_mode': args.compile
        },
        'training_config': {
            'optimizer': args.opt,
            'learning_rate': args.lr,
            'weight_decay': args.wd,
            'momentum': args.momentum,
            'scheduler': args.scheduler,
            'lr_steps': args.lr_steps,
            'gamma': args.gamma,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'meta_batch_size': args.meta_batch_size,
            'grad_clip': args.grad_clip,
            'amp': args.amp,
            'label_smoothing': 0.0,  # Fixed for language models
            'predparam_wd': 0 if args.ghn2 else 3e-5
        },
        'data_config': {
            'dataset': 'WikiText-2',
            'vocab_size': args.vocab_size,
            'seq_len': args.seq_len,
            'tokenizer': 'gpt2',
            'num_workers': args.num_workers
        },
        'hardware_config': {
            'device': args.device,
            'ddp': ddp.ddp if 'ddp' in globals() else False,
            'world_size': ddp.world_size if 'ddp' in globals() and ddp.ddp else 1,
            'rank': ddp.rank if 'ddp' in globals() else 0
        },
        'paths': {
            'save_dir': args.save_dir,
            'data_dir': args.data_dir,
            'experiment_dir': f'experiment_{run_id}_{args.name}',
            'tensorboard_logs_dir': 'tensorboard_logs',
            'tensorboard_logs_location': os.path.join(args.save_dir, 'tensorboard_logs'),
            'best_model_dir': f'experiment_{run_id}_{args.name}/best_model',
            'checkpoints_dir': f'experiment_{run_id}_{args.name}/checkpoints',
            'metadata_dir': f'experiment_{run_id}_{args.name}/metadata'
        }
    }
    return metadata


def create_training_visualization_script(metrics_file, save_dir):
    """Create a Python script to visualize training metrics."""
    viz_script = f'''#!/usr/bin/env python3
"""
Training metrics visualization script.
Generated automatically by train_ghn_ddp.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_metrics(metrics_file):
    """Plot training metrics from the saved JSON file."""
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    train_perplexity = metrics['train_perplexity']
    learning_rate = metrics['learning_rate']
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.axhline(y=metrics['best_loss'], color='r', linestyle='--', 
                label=f'Best Loss: {{metrics["best_loss"]:.4f}} (Epoch {{metrics["best_epoch"]}})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training perplexity
    ax2.plot(epochs, train_perplexity, 'g-', linewidth=2, label='Training Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax3.plot(epochs, learning_rate, 'orange', linewidth=2, label='Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot loss vs perplexity
    ax4.scatter(train_loss, train_perplexity, c=epochs, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Training Loss')
    ax4.set_ylabel('Training Perplexity')
    ax4.set_title('Loss vs Perplexity')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(metrics_file), 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Training metrics plot saved to: {{plot_path}}')
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\\n=== Training Summary ===")
    print(f"Total epochs: {{len(epochs)}}")
    print(f"Best loss: {{metrics['best_loss']:.4f}} (Epoch {{metrics['best_epoch']}})")
    print(f"Final loss: {{train_loss[-1]:.4f}}")
    print(f"Final perplexity: {{train_perplexity[-1]:.2f}}")
    print(f"Loss improvement: {{train_loss[0] - train_loss[-1]:.4f}}")
    print(f"Perplexity improvement: {{train_perplexity[0] - train_perplexity[-1]:.2f}}")

if __name__ == "__main__":
    metrics_file = "{metrics_file}"
    plot_training_metrics(metrics_file)
'''
    
    # Save the visualization script
    viz_script_path = os.path.join(save_dir, 'plot_training_metrics.py')
    with open(viz_script_path, 'w') as f:
        f.write(viz_script)
    
    # Make it executable
    os.chmod(viz_script_path, 0o755)
    
    print(f'Training visualization script created: {viz_script_path}')
    print('Run: python plot_training_metrics.py to generate training plots')


def main():
    parser = argparse.ArgumentParser(description='GHN-3 training for Language Models')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')
    parser.add_argument('--vocab_size', type=int, default=50257, help='vocabulary size for language models')
    parser.add_argument('--seq_len', type=int, default=256, help='sequence length for language models')
    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       layers=3,  # default number of layers in GHN-3
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)
    
    # Override image-based configs for language model training
    args.dataset = 'language_model'  # Override cifar10
    args.imsize = None  # Not applicable for language models
    # Set max_shape to handle the largest weights in language models
    # Note: GHN predicts parameters in chunks and tiles them for large embeddings
    # Start with a conservative size based on the original working configuration
    # Can be increased gradually if needed for larger models
    args.max_shape = (args.seq_len, args.seq_len, 8, 8)  # (64, 64, 8, 8) for seq_len=64

    if hasattr(args, 'multigpu') and args.multigpu:
        raise NotImplementedError(
            'the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. '
            'nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel '
            '(https://github.com/pytorch/pytorch/issues/659360).'
            'Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. '
            'nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top).')

    log('loading language model architectures and WikiText-2 dataset...')
    
    # Load language model architectures
    arch_loader, arch_configs = build_ghn_variants_dataloader(
        batch_size=args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
        vocab_size=args.vocab_size,
        max_len=max(args.seq_len * 2, 1024),  # Ensure max_len is at least 2x seq_len
        device=args.device,
        num_workers=args.num_workers,
        ve_cutoff=args.virtual_edges,
        dense=True  # GHN-3 requires dense graphs
    )
    
    # Load WikiText-2 dataset
    wt2_data = build_wikitext2(
        tokenizer_name="gpt2",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.data_dir
    )
    
    train_queue = wt2_data["train_loader"]
    num_classes = args.vocab_size  # For language models, vocab_size is the number of classes

    hid = args.hid
    # For language models, we use a different max_shape calculation
    # Based on typical transformer dimensions
    s = 16  # Default sequence dimension for language models
    default_max_shape = (hid * 2, hid * 2, s, s) if ghn2 else (hid, hid, s, s)
    log('current max_shape: {} {} default max_shape: {}'.format(args.max_shape,
                                                                '=' if args.max_shape == default_max_shape else '!=',
                                                                default_max_shape))

    config = {'max_shape': args.max_shape, 'num_classes': num_classes, 'hid': hid, 
              'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2}

    ghn = GHN3(**config, debug_level=args.debug)
    
    # Use language model architecture loader instead of DeepNets1M
    graphs_queue = arch_loader

    trainer = Trainer(ghn,
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd, 'momentum': args.momentum},
                      scheduler='mstep' if args.scheduler is None else args.scheduler,
                      scheduler_args={'milestones': args.lr_steps, 'gamma': args.gamma},
                      n_batches=len(train_queue),
                      grad_clip=min(args.grad_clip, 1.0),  # Cap gradient clipping for stability
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0 if ghn2 else 1e-5,  # Reduced parameter regularization
                      label_smoothing=0.0,  # No label smoothing for language models
                      save_dir=args.save,
                      ckpt=args.ckpt,
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      compile_mode=args.compile)

    log('\nStarting training GHN with {} parameters!'.format(sum([p.numel() for p in ghn.parameters()])))
    log(f'Number of architecture variants: {len(arch_configs)}')
    
    # Efficiency recommendation: Use ghn3.ops as base classes during training
    log('Note: For efficiency, it is recommended to use ghn3.ops as base classes during training the GHN')
    log('Note: Perplexity overflow protection implemented - loss clamped to max 10.0 to prevent exp() overflow')
    log('Note: Additional stability fixes: reduced grad_clip, reduced predparam_wd, improved AMP settings')
    
    # Initialize iterators
    graphs_queue = iter(graphs_queue)
    token_iter = iter(train_queue)

    # Create experiment tracking
    start_time = datetime.now()
    run_id = str(uuid.uuid4())[:8]  # Short unique ID for this run
    
    # Initialize metrics tracking and best model saving
    training_metrics = {
        'epochs': [],
        'train_loss': [],
        'train_perplexity': [],
        'learning_rate': [],
        'best_loss': float('inf'),
        'best_epoch': 0,
        'run_id': run_id,
        'start_time': start_time.isoformat()
    }
    
    # Create organized directory structure for saving models and metadata
    experiment_dir = os.path.join(args.save_dir, f'experiment_{run_id}_{args.name}')
    best_model_dir = os.path.join(experiment_dir, 'best_model')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    metadata_dir = os.path.join(experiment_dir, 'metadata')
    
    # Create all directories
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # File paths
    metrics_file = os.path.join(metadata_dir, 'training_metrics.json')
    metadata_file = os.path.join(metadata_dir, 'experiment_metadata.json')
    
    # Create experiment metadata
    experiment_metadata = create_experiment_metadata(args, config, run_id, start_time)
    
    # Initialize TensorBoard writer with experiment tracking (only on rank 0 to avoid conflicts)
    tensorboard_dir = os.path.join(args.save_dir, 'tensorboard_logs')
    writer = None
    if ddp.rank == 0:
        os.makedirs(tensorboard_dir, exist_ok=True)
        # Save TensorBoard logs directly in tensorboard_logs folder
        writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Save experiment metadata
        with open(metadata_file, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Log hyperparameters to TensorBoard
        writer.add_hparams(
            {
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'hidden_dim': args.hid,
                'num_layers': args.layers,
                'num_heads': args.heads,
                'optimizer': args.opt,
                'scheduler': args.scheduler,
                'vocab_size': args.vocab_size,
                'seq_len': args.seq_len,
                'epochs': args.epochs,
                'amp': args.amp
            },
            {}
        )
        
        log(f'Experiment ID: {run_id}')
        log(f'Experiment directory: {experiment_dir}')
        log(f'TensorBoard logs will be saved to: {tensorboard_dir}')
        log(f'Experiment metadata saved to: {metadata_file}')
        log(f'Best models will be saved to: {best_model_dir}')
        log(f'Periodic checkpoints will be saved to: {checkpoints_dir}')
        log('To view TensorBoard, run: tensorboard --logdir=' + tensorboard_dir)

    for epoch in range(trainer.start_epoch, args.epochs):
        log('\n=== EPOCH {}/{} ==='.format(epoch + 1, args.epochs))
        log('Learning Rate: {:.2e}'.format(trainer.get_lr()))
        log('Training Loop: For each token batch -> For each model architecture -> GHN-3 predicts parameters -> Forward pass -> Compute loss -> Backward pass -> Log metrics')

        trainer.reset_metrics(epoch)

        # Create progress bar for training steps
        pbar = tqdm(range(len(train_queue)), 
                   desc=f'Epoch {epoch+1}/{args.epochs}', 
                   unit='batch',
                   disable=ddp.rank != 0)  # Only show progress bar on rank 0

        for step in pbar:
            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break

            # ===== TRAINING LOOP STRUCTURE =====
            # For each token batch, run all model architectures
            
            # 1. Get next batch of WikiText data
            try:
                token_batch = next(token_iter)
            except StopIteration:
                token_iter = iter(wt2_data["train_loader"])
                token_batch = next(token_iter)

            # Extract input_ids and labels for language modeling
            input_ids = token_batch["input_ids"]
            labels = token_batch["labels"]
            
            # 2. For this token batch, run through all model architectures
            # Reset the architecture iterator for this token batch
            arch_iter = iter(arch_loader)
            
            for arch_step in range(len(arch_loader)):
                try:
                    models, graph_batch, metas = next(arch_iter)
                except StopIteration:
                    break
                
                # 3. GHN-3 predicts parameters for this architecture
                # 4. Forward pass through predicted model on this token batch
                # 5. Compute loss (cross-entropy + parameter regularization)
                # 6. Backward pass and optimizer step
                # 7. Log metrics
                # (All handled by trainer.update_lm)
                trainer.update_lm(input_ids, labels, graphs=graph_batch, models=models)
            
            # Update progress bar with current metrics
            if ddp.rank == 0:  # Only update progress bar on rank 0
                pbar.set_postfix({
                    'loss': f'{trainer.metrics["loss"].avg:.4f}',
                    'perplexity': f'{trainer.metrics["top1"].avg:.2f}',
                    'lr': f'{trainer.get_lr():.2e}'
                })
                
                # Log metrics to TensorBoard at regular intervals
                if step % args.log_interval == 0 and writer is not None:
                    global_step = epoch * len(train_queue) + step
                    writer.add_scalar('Training/Loss', trainer.metrics["loss"].avg, global_step)
                    writer.add_scalar('Training/Perplexity', trainer.metrics["top1"].avg, global_step)
                    writer.add_scalar('Training/Learning_Rate', trainer.get_lr(), global_step)
            
            trainer.log(step)

            if args.save:
                # save GHN checkpoint
                trainer.save(epoch, step, {'args': args, 'config': config}, interm_epoch=args.interm_epoch)
        
        # Close progress bar
        pbar.close()

        # Collect epoch metrics
        epoch_loss = trainer.metrics['loss'].avg
        epoch_perplexity = trainer.metrics['top1'].avg
        current_lr = trainer.get_lr()
        
        # Store metrics
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['train_loss'].append(float(epoch_loss))
        training_metrics['train_perplexity'].append(float(epoch_perplexity))
        training_metrics['learning_rate'].append(float(current_lr))
        
        # Log epoch-level metrics to TensorBoard
        if ddp.rank == 0 and writer is not None:
            writer.add_scalar('Epoch/Loss', epoch_loss, epoch + 1)
            writer.add_scalar('Epoch/Perplexity', epoch_perplexity, epoch + 1)
            writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch + 1)
        
        # Check if this is the best model so far
        if epoch_loss < training_metrics['best_loss']:
            training_metrics['best_loss'] = float(epoch_loss)
            training_metrics['best_epoch'] = epoch + 1
            
            # Save best model (only on rank 0 to avoid conflicts)
            if ddp.rank == 0:
                best_model_path = os.path.join(best_model_dir, 'best_ghn_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': ghn.state_dict(),
                    'optimizer_state_dict': trainer._optimizer.state_dict(),
                    'scheduler_state_dict': trainer._scheduler.state_dict() if trainer._scheduler else None,
                    'loss': epoch_loss,
                    'perplexity': epoch_perplexity,
                    'config': config,
                    'args': args
                }, best_model_path)
                log(f'New best model saved at epoch {epoch + 1} with loss {epoch_loss:.4f}')
                
                # Log best model metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar('Best_Model/Loss', epoch_loss, epoch + 1)
                    writer.add_scalar('Best_Model/Perplexity', epoch_perplexity, epoch + 1)
        
        # Save metrics to file (only on rank 0)
        if ddp.rank == 0:
            with open(metrics_file, 'w') as f:
                json.dump(training_metrics, f, indent=2)
        
        log(f'Epoch {epoch + 1} completed - Loss: {epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}, LR: {current_lr:.2e}')
        
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 and ddp.rank == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ghn.state_dict(),
                'optimizer_state_dict': trainer._optimizer.state_dict(),
                'scheduler_state_dict': trainer._scheduler.state_dict() if trainer._scheduler else None,
                'loss': epoch_loss,
                'perplexity': epoch_perplexity,
                'config': config,
                'args': args,
                'training_metrics': training_metrics
            }, checkpoint_path)
            log(f'Periodic checkpoint saved at epoch {epoch + 1}: {checkpoint_path}')
        
        trainer.scheduler_step()  # lr scheduler step

    # Final training summary
    if ddp.rank == 0:
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Update experiment metadata with final results
        experiment_metadata['experiment_info']['end_time'] = end_time.isoformat()
        experiment_metadata['experiment_info']['duration_seconds'] = duration.total_seconds()
        experiment_metadata['results'] = {
            'best_loss': training_metrics["best_loss"],
            'best_epoch': training_metrics["best_epoch"],
            'final_loss': training_metrics["train_loss"][-1],
            'final_perplexity': training_metrics["train_perplexity"][-1],
            'total_epochs_completed': len(training_metrics["epochs"])
        }
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        log('\n=== Training Summary ===')
        log(f'Experiment ID: {run_id}')
        log(f'Duration: {duration}')
        log(f'Best model achieved at epoch {training_metrics["best_epoch"]} with loss {training_metrics["best_loss"]:.4f}')
        log(f'Final training loss: {training_metrics["train_loss"][-1]:.4f}')
        log(f'Final training perplexity: {training_metrics["train_perplexity"][-1]:.2f}')
        log(f'Best model saved to: {os.path.join(best_model_dir, "best_ghn_model.pt")}')
        log(f'Training metrics saved to: {metrics_file}')
        log(f'Experiment metadata saved to: {metadata_file}')
        log(f'All experiment files saved in: {experiment_dir}')
        
        # Create training visualization script
        create_training_visualization_script(metrics_file, args.save_dir)
        
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
            log('TensorBoard writer closed')
    
    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()
