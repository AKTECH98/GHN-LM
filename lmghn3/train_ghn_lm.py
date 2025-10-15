
import argparse
import time
import torch
from functools import partial

from ppuda.config import init_config
from CustomGHN3 import GHN3, log, Trainer, setup_ddp, clean_ddp
from lmghn3.language_models.lm_arch_loader import build_ghn_variants_dataloader
from lmghn3.language_models.wikitext2_loader import build_wikitext2

log = partial(log, flush=True)


def main():
    parser = argparse.ArgumentParser(description='GHN-3 training for Language Models')
    paser.add_argument("--max_shape", type=tuple, default=(1024, 1024, 1, 1), help='max shape for the GHN-3')
    parser.add_argument("--seq_len", type=int, default=256, help='sequence length for the language model')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')
    
    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       layers=3,  # default number of layers in GHN-3
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)
    

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
        vocab_size=args.vocab_size,
        max_len=max(args.seq_len * 2, 1024),  # Ensure max_len is at least 2x seq_len
        device=args.device,
        num_workers=args.num_workers,
        ve_cutoff=args.virtual_edges,
        dense=True  # GHN-3 requires dense graphs
    )
    
    

    hid = args.hid

    config = {'max_shape': args.max_shape, 'num_classes': num_classes, 'hypernet': args.hypernet,
              'decoder': args.decoder, 'weight_norm': args.weight_norm, 've': args.virtual_edges > 1,
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2}

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
             
            trainer.log(step)

            if args.save:
                trainer.save(epoch, step, {'args': args, 'config': config}, interm_epoch=args.interm_epoch)
    
    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()
