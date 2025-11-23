
import argparse
import time
import torch
import warnings
import os
import json
import multiprocessing
from functools import partial
from torch.utils.tensorboard import SummaryWriter

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This is required when using num_workers > 0 with CUDA in DataLoader
# Must be set before any DataLoader is created
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

# Suppress NetworkX backend warning
warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")

from ppuda.config import init_config
from GHN import GHN3, log, Trainer, setup_ddp, clean_ddp
from Dataloader.lm_arch_loader import build_lm_variants_dataloader
from Dataloader.wikitext2_loader import build_wikitext2

log = partial(log, flush=True)


def parse_arguments():
    """Parse command line arguments for GHN training."""
    parser = argparse.ArgumentParser(description='GHN-3 training for Language Models')

    parser.add_argument("--model_name", type=str, default=None, help='name of the model')

    # GHN-specific arguments
    parser.add_argument("--seq_len", type=int, default=256, help='sequence length for the language model')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')
    parser.add_argument('--include_embeddings', action='store_true', help='include embedding layers in GHN prediction (default: exclude embeddings)')
    parser.add_argument('--max_params', type=float, default=None, help='Maximum number of parameters (in billions, e.g., 4.0 for 4B). Filters out larger models to avoid OOM. Recommended: 4.0-5.0 for 40GB GPU.')
    
    
    # PPUDA-specific arguments (common training args handled by init_config)
    
    # Model architecture arguments
    
    # Data arguments (handled by init_config)
    
    # System arguments (handled by init_config)
    
    return parser


def main():
    # Parse arguments
    parser = parse_arguments()
    parsed_args = parser.parse_known_args()[0]
    ghn2 = parsed_args.ghn2

    ddp = setup_ddp()
    # debug=0: Fast training (default)
    # debug=1: Basic parameter prediction info
    # debug=2: Warnings about parameter mismatches
    # debug=3: Full parameter statistics (min/max/mean/std/norm for each parameter) - VERY VERBOSE, use only for debugging
    # Note: To enable debug_level > 2, you can override this or use --debug flag if supported
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)
    
    # Get job ID from environment or generate one
    job_id = args.model_name #os.environ.get('SLURM_JOB_ID', f'ghn3_lm_{int(time.time())}')
    
    # Create directory structure
    logging_dir = 'tensor_log'
    experiment_dir = 'Experiment'
    job_experiment_dir = os.path.join(experiment_dir, job_id)
    
    # Create directories
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(job_experiment_dir, exist_ok=True)
    
    # Initialize TensorBoard writer (only on rank 0)
    tensorboard_writer = None
    if ddp.rank == 0:
        tensorboard_log_dir = os.path.join(logging_dir, job_id)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        log(f'TensorBoard logs will be saved to: {tensorboard_log_dir}')
        log(f'Experiment data will be saved to: {job_experiment_dir}')
    

    args.dataset = 'WikiText-2'

    if hasattr(args, 'multigpu') and args.multigpu:
        raise NotImplementedError(
            'the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. '
            'nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel '
            '(https://github.com/pytorch/pytorch/issues/659360).'
            'Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. '
            'nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top).')

    log('loading the %s dataset...' % args.dataset)
    
    # Load WikiText-2 dataset
    wt2_data = build_wikitext2(
        tokenizer_name="gpt2",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.data_dir
    )
    
    train_queue = wt2_data["train_loader"]

    # For language models, vocab_size is the number of classes
    num_classes = wt2_data["vocab_size"]  

    # Load language model architectures
    arch_loader, arch_configs = build_lm_variants_dataloader(
        batch_size=args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
        vocab_size=num_classes,
        max_len=args.seq_len,  # Match model max_seq_len to training seq_len
        device=args.device,
        num_workers=args.num_workers,
        ve_cutoff=args.virtual_edges,
        dense=True,  # GHN-3 requires dense graphs
        exclude_embeddings=not args.include_embeddings,  # Convert include_embeddings to exclude_embeddings
        max_params=int(args.max_params * 1e9) if args.max_params is not None else None,  # Convert billions to actual count
    )
    
    log(f'Loaded {len(arch_configs):,} language model variants for training')
    
    

    hid = args.hid

    config = {'num_classes': num_classes, 'hypernet': args.hypernet,
              'decoder': args.decoder, 'weight_norm': True, 've': args.virtual_edges > 1,  # Force weight_norm=True for language models
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2,
              'exclude_embeddings': not args.include_embeddings, 'max_shape': args.max_shape,
              'lm_scale_factor': 1.0}  # Not used - using original GHN-3 normalization with gentle embedding rescale

    ghn = GHN3(**config, debug_level=args.debug)
    
    # Log the actual weight_norm value being used (may differ from args.weight_norm)
    if ddp.rank == 0:
        log(f'GHN created with weight_norm={ghn.weight_norm} (forced to True for language models)')
        log(f'Using original GHN-3 normalization with gentle embedding rescale (std=0.02)')
    
    # Save config metadata to experiment directory
    if ddp.rank == 0:
        # Organize config into logical sections
        ghn_config = {
            'hypernet': args.hypernet,
            'decoder': args.decoder,
            'weight_norm': True,  # Forced to True for language models (required for normalization)
            'virtual_edges': args.virtual_edges,
            'layernorm': args.ln,
            'hid': args.hid,
            'layers': args.layers,
            'heads': args.heads,
            'is_ghn2': ghn2,
            'exclude_embeddings': not args.include_embeddings,
            'max_shape': args.max_shape,
            'normalization': 'original_ghn3_with_embedding_rescale_0.02'  # Using original GHN-3 fan-in normalization with gentle embedding rescale
        }
        
        training_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'meta_batch_size': args.meta_batch_size,
            'lr': args.lr,
            'wd': args.wd,
            'momentum': args.momentum,
            'opt': args.opt,
            'scheduler': args.scheduler,
            'lr_steps': args.lr_steps,
            'gamma': args.gamma,
            'grad_clip': args.grad_clip,
            'amp': args.amp,
            'log_interval': args.log_interval,
            'interm_epoch': args.interm_epoch,
            'compile': args.compile
        }
        
        dataset_config = {
            'name': args.dataset,
            'num_classes': num_classes,
            'seq_len': args.seq_len,
            'data_dir': args.data_dir,
            'num_workers': args.num_workers
        }
        
        config_metadata = {
            'job_id': job_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ghn_config': ghn_config,
            'training_config': training_config,
            'dataset_config': dataset_config
        }
        
        config_path = os.path.join(job_experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_metadata, f, indent=2)
        log(f'Config metadata saved to: {config_path}')
    
    # Use language model architecture loader instead of DeepNets1M
    graphs_queue = arch_loader

    # Update save directory to experiment folder (always use experiment directory)
    args.save = job_experiment_dir
    log(f'Checkpoints will be saved to: {job_experiment_dir}')
    
    trainer = Trainer(ghn,
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd, 'momentum': args.momentum},
                      scheduler='cosine' if args.scheduler is None else args.scheduler,
                      scheduler_args={'milestones': [int(args.epochs * 0.3), int(args.epochs * 0.6)], 'gamma': args.gamma} if args.scheduler == 'mstep' else {},
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,  # Use original gradient clipping value
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0,  # Turned off predicted-parameter regularization at first
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
    log('Note: Using original GHN-3 normalization with gentle embedding rescale (std=0.02)')
    log('Note: Loss is NOT clamped before backward() - only perplexity calculation uses clamping for overflow protection')
    log('Note: Meta-learning may need several epochs to show progress - monitor loss curve over 5+ epochs before concluding it is stuck')
    log('Note: Additional stability fixes: reduced grad_clip, reduced predparam_wd, improved AMP settings')
    
    # Initialize iterators
    graphs_queue = iter(graphs_queue)
    
    # Track best metrics for saving best model
    best_loss = float('inf')
    best_perplexity = float('inf')

    for epoch in range(trainer.start_epoch, args.epochs):
        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        trainer.reset_metrics(epoch)

        for step, token_batch in enumerate(train_queue, start=trainer.start_step):
            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break

            # Extract input_ids and labels from the token batch
            input_ids = token_batch["input_ids"]
            labels = token_batch["labels"]

            # Get next batch of model architectures
            try:
                models, graph_batch, metas = next(graphs_queue)
            except StopIteration:
                graphs_queue = iter(arch_loader)
                models, graph_batch, metas = next(graphs_queue)

            # Move models to GPU if they were created on CPU (for multiprocessing workers)
            # This avoids CUDA OOM in worker processes
            if args.device == 'cuda':
                models = [model.to(args.device) if next(model.parameters()).device.type == 'cpu' else model 
                         for model in models]

            # Update trainer with language model data
            trainer.update_lm(input_ids, labels, graphs=graph_batch, models=models)
            
            # Step the scheduler after each batch (only for step-based schedulers like cosine-warmup)
            trainer.scheduler_step()
            
            # Explicit cleanup: Models are created lazily and should be freed after use
            # Python's garbage collector will handle this, but explicit cleanup helps with memory
            del models, graph_batch
            if step % 10 == 0:  # Periodic cleanup to avoid fragmentation
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
             
            # Log to console
            trainer.log(step)
            
            # Log to TensorBoard (only on rank 0)
            if tensorboard_writer is not None and step % args.log_interval == 0:
                metrics = {metric: value.avg for metric, value in trainer.metrics.items()}
                
                # Log loss and perplexity
                tensorboard_writer.add_scalar('Train/Loss', metrics.get('loss', 0), step)
                if 'perplexity' in metrics:
                    ppl = metrics['perplexity']
                    # Handle inf perplexity
                    if ppl == float('inf'):
                        tensorboard_writer.add_scalar('Train/Perplexity', 22032.0, step)  # Use clamped max for visualization
                    else:
                        tensorboard_writer.add_scalar('Train/Perplexity', ppl, step)
                
                if 'loss_predwd' in metrics:
                    tensorboard_writer.add_scalar('Train/Loss_PredWD', metrics['loss_predwd'], step)
                
                # Log learning rate
                tensorboard_writer.add_scalar('Train/LR', trainer.get_lr(), step)
                
                # Log epoch
                tensorboard_writer.add_scalar('Train/Epoch', epoch, step)

            if args.save:
                trainer.save(epoch, step, {'args': args, 'config': config}, interm_epoch=args.interm_epoch)
                
                # Save best model based on loss
                current_loss = trainer.metrics['loss'].avg
                perplexity_meter = trainer.metrics.get('perplexity')
                if perplexity_meter is None:
                    current_perplexity = float('inf')
                else:
                    current_perplexity = perplexity_meter.avg
                
                if current_perplexity == float('inf'):
                    current_perplexity_display = 'inf'
                else:
                    current_perplexity_display = f'{current_perplexity:.4f}'
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model_path = os.path.join(job_experiment_dir, 'best_model.pt')
                    torch.save({
                        'state_dict': (trainer._model.module if hasattr(trainer._model, 'module') 
                                      else trainer._model).state_dict(),
                        'optimizer': trainer._optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'loss': current_loss,
                        'perplexity': current_perplexity if current_perplexity != float('inf') else 22032.0,
                        'args': args,
                        'config': config
                    }, best_model_path)
                    log(f'New best model saved with loss: {current_loss:.4f}, perplexity: {current_perplexity_display}')
        
        # Log epoch-level metrics to TensorBoard
        if tensorboard_writer is not None:
            epoch_metrics = {metric: value.avg for metric, value in trainer.metrics.items()}
            tensorboard_writer.add_scalar('Epoch/Loss', epoch_metrics.get('loss', 0), epoch)
            if 'perplexity' in epoch_metrics:
                ppl = epoch_metrics['perplexity']
                if ppl == float('inf'):
                    tensorboard_writer.add_scalar('Epoch/Perplexity', 22032.0, epoch)  # Use clamped max for visualization
                else:
                    tensorboard_writer.add_scalar('Epoch/Perplexity', ppl, epoch)
            tensorboard_writer.add_scalar('Epoch/LR', trainer.get_lr(), epoch)
        
        # Step the scheduler after each epoch (only for epoch-based schedulers like cosine, step, mstep)
        trainer.scheduler_step_epoch()
    
    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    
    # Close TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
        log(f'TensorBoard logs saved to: {tensorboard_log_dir}')
        log(f'Best model saved to: {os.path.join(job_experiment_dir, "best_model.pt")}')
    
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()
