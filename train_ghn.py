
import argparse
import time
import torch
import warnings
import os
import json
from functools import partial
from torch.utils.tensorboard import SummaryWriter

# Suppress NetworkX backend warning
warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")

from ppuda.config import init_config
from GHN import GHN3, log, Trainer, setup_ddp, clean_ddp
from Dataloader.lm_arch_loader import build_ghn_variants_dataloader
from Dataloader.wikitext2_loader import build_wikitext2

log = partial(log, flush=True)


def main():
    parser = argparse.ArgumentParser(description='GHN-3 training for Language Models')
    parser.add_argument("--seq_len", type=int, default=256, help='sequence length for the language model')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')
    parser.add_argument('--include_embeddings', action='store_true', help='include embedding layers in GHN prediction (default: exclude embeddings)')
    
    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       layers=3,  # default number of layers in GHN-3
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)
    
    # Get job ID from environment or generate one
    job_id = os.environ.get('SLURM_JOB_ID', f'ghn3_lm_{int(time.time())}')
    
    # Create directory structure
    logging_dir = 'logging'
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
    arch_loader, arch_configs = build_ghn_variants_dataloader(
        batch_size=args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
        vocab_size=num_classes,
        max_len=max(args.seq_len * 2, 1024),  # Ensure max_len is at least 2x seq_len
        device=args.device,
        num_workers=args.num_workers,
        ve_cutoff=args.virtual_edges,
        dense=True,  # GHN-3 requires dense graphs
        exclude_embeddings=not args.include_embeddings  # Convert include_embeddings to exclude_embeddings
    )
    
    

    hid = args.hid

    config = {'max_shape': (1024, 1024, 1, 1), 'num_classes': num_classes, 'hypernet': args.hypernet,
              'decoder': args.decoder, 'weight_norm': args.weight_norm, 've': args.virtual_edges > 1,
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2,
              'exclude_embeddings': not args.include_embeddings}

    ghn = GHN3(**config, debug_level=args.debug)
    
    # Save config metadata to experiment directory
    if ddp.rank == 0:
        config_metadata = {
            'job_id': job_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'args': vars(args),
            'config': config,
            'dataset': args.dataset,
            'num_classes': num_classes,
            'seq_len': args.seq_len,
            'heads': args.heads,
            'include_embeddings': args.include_embeddings,
            'ghn2': ghn2
        }
        
        config_path = os.path.join(job_experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_metadata, f, indent=2)
        log(f'Config metadata saved to: {config_path}')
    
    # Use language model architecture loader instead of DeepNets1M
    graphs_queue = arch_loader

    # Update save directory to experiment folder
    if args.save:
        args.save = job_experiment_dir
    
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

            # Update trainer with language model data
            trainer.update_lm(input_ids, labels, graphs=graph_batch, models=models)
             
            # Log to console
            trainer.log(step)
            
            # Log to TensorBoard (only on rank 0)
            if tensorboard_writer is not None and step % args.log_interval == 0:
                metrics = {metric: value.avg for metric, value in trainer.metrics.items()}
                
                # Log loss and perplexity
                tensorboard_writer.add_scalar('Train/Loss', metrics.get('loss', 0), step)
                tensorboard_writer.add_scalar('Train/Perplexity', metrics.get('top1', 0), step)  # top1 stores perplexity for LM
                
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
                current_perplexity = trainer.metrics['top1'].avg  # top1 stores perplexity for LM
                
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
                        'perplexity': current_perplexity,
                        'args': args,
                        'config': config
                    }, best_model_path)
                    log(f'New best model saved with loss: {current_loss:.4f}, perplexity: {current_perplexity:.4f}')
        
        # Log epoch-level metrics to TensorBoard
        if tensorboard_writer is not None:
            epoch_metrics = {metric: value.avg for metric, value in trainer.metrics.items()}
            tensorboard_writer.add_scalar('Epoch/Loss', epoch_metrics.get('loss', 0), epoch)
            tensorboard_writer.add_scalar('Epoch/Perplexity', epoch_metrics.get('top1', 0), epoch)
            tensorboard_writer.add_scalar('Epoch/LR', trainer.get_lr(), epoch)
    
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
