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
    args.max_shape = (args.seq_len, args.seq_len, 8, 8)  # Set appropriate max_shape for language models

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
                      grad_clip=args.grad_clip,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0 if ghn2 else 3e-5,
                      label_smoothing=0.0,  # No label smoothing for language models
                      save_dir=args.save,
                      ckpt=args.ckpt,
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      compile_mode=args.compile)

    log('\nStarting training GHN with {} parameters!'.format(sum([p.numel() for p in ghn.parameters()])))
    log(f'Number of architecture variants: {len(arch_configs)}')
    
    # Initialize iterators
    graphs_queue = iter(graphs_queue)
    token_iter = iter(train_queue)

    # Initialize metrics tracking and best model saving
    training_metrics = {
        'epochs': [],
        'train_loss': [],
        'train_perplexity': [],
        'learning_rate': [],
        'best_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Create directory for saving best model and metrics
    best_model_dir = os.path.join(args.save_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    metrics_file = os.path.join(args.save_dir, 'training_metrics.json')

    for epoch in range(trainer.start_epoch, args.epochs):

        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        trainer.reset_metrics(epoch)

        # Create progress bar for training steps
        pbar = tqdm(range(len(train_queue)), 
                   desc=f'Epoch {epoch+1}/{args.epochs}', 
                   unit='batch',
                   disable=ddp.rank != 0)  # Only show progress bar on rank 0

        for step in pbar:

            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break

            # Get next architecture batch
            try:
                models, graph_batch, metas = next(graphs_queue)
            except StopIteration:
                graphs_queue = iter(arch_loader)
                models, graph_batch, metas = next(graphs_queue)
            
            # Get next token batch
            try:
                token_batch = next(token_iter)
            except StopIteration:
                token_iter = iter(wt2_data["train_loader"])
                token_batch = next(token_iter)

            # Extract input_ids and labels for language modeling
            input_ids = token_batch["input_ids"]
            labels = token_batch["labels"]
            
            # Update trainer with language model data
            trainer.update_lm(input_ids, labels, graphs=graph_batch, models=models)
            
            # Update progress bar with current metrics
            if ddp.rank == 0:  # Only update progress bar on rank 0
                pbar.set_postfix({
                    'loss': f'{trainer.metrics["loss"].avg:.4f}',
                    'perplexity': f'{trainer.metrics["top1"].avg:.2f}',
                    'lr': f'{trainer.get_lr():.2e}'
                })
            
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
        
        # Save metrics to file (only on rank 0)
        if ddp.rank == 0:
            with open(metrics_file, 'w') as f:
                json.dump(training_metrics, f, indent=2)
        
        log(f'Epoch {epoch + 1} completed - Loss: {epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}, LR: {current_lr:.2e}')
        
        trainer.scheduler_step()  # lr scheduler step

    # Final training summary
    if ddp.rank == 0:
        log('\n=== Training Summary ===')
        log(f'Best model achieved at epoch {training_metrics["best_epoch"]} with loss {training_metrics["best_loss"]:.4f}')
        log(f'Final training loss: {training_metrics["train_loss"][-1]:.4f}')
        log(f'Final training perplexity: {training_metrics["train_perplexity"][-1]:.2f}')
        log(f'Best model saved to: {os.path.join(best_model_dir, "best_ghn_model.pt")}')
        log(f'Training metrics saved to: {metrics_file}')
        
        # Create training visualization script
        create_training_visualization_script(metrics_file, args.save_dir)
    
    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()
