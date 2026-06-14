# Copyright (c) 2023. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper to train models.

"""


import os
import math
import torch
import torch.nn as nn
import numpy as np
import psutil
import traceback
import torch.distributed as dist
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, LambdaLR
from ppuda.utils import AvgrageMeter, accuracy, capacity, init
from ppuda.ghn.nn import GHN
from .utils import Logger, print_grads, log
from .ddp_utils import is_ddp, get_ddp_rank, avg_ddp_metric
from .nn import from_pretrained
from .ops import Network

try:
    from timm.optim import Lamb  # timm was not used in the paper's experiments, so it's optional
    from timm.loss import BinaryCrossEntropy
    from timm.data.mixup import Mixup
except Exception as e:
    print(e)

log = partial(log, flush=True)
process = psutil.Process(os.getpid())


class Trainer:
    def __init__(self,
                 model,
                 opt,
                 opt_args,
                 scheduler,
                 n_batches,
                 grad_clip=5,
                 auxiliary=False,
                 auxiliary_weight=0.4,
                 device='cuda',
                 log_interval=100,
                 label_smoothing=0,  # default value for ImageNet
                 predparam_wd=0,  # our predicted parameter regularization
                 scheduler_args=None,
                 save_dir=None,
                 ckpt=None,
                 epochs=None,
                 verbose=False,
                 amp=False,
                 amp_min_scale=None,            # 1024 for GHN-3
                 amp_growth_interval=2000,      # 100 for GHN-3
                 bce=False,
                 mixup=False,
                 compile_mode=None,
                 beta=1e-5,  # to add noise to predicted parameters to improve their fine-tuning
                 ):

        if bce:
            self.criterion = BinaryCrossEntropy(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.n_batches = n_batches
        self.grad_clip = grad_clip
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.device = device
        self.log_interval = log_interval
        self.amp = amp
        self.amp_min_scale = amp_min_scale
        self.predparam_wd = predparam_wd
        self.epochs = epochs
        self.verbose = verbose
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler(growth_interval=amp_growth_interval)
        self.ddp = is_ddp()
        if self.ddp:
            self.rank = get_ddp_rank()
            if self.verbose:
                print(f'trainer rank {self.rank}')
        else:
            self.rank = device

        self.mixup_fn = Mixup(mixup_alpha=0.1, cutmix_alpha=1.0) if mixup else None
        if predparam_wd > 0:
            self.param_decay = lambda p: torch.norm(p, p='fro')

        model.to(self.rank)

        # Automatically resume from a checkpoint if exists or use GHN to initialize the model if ckpt is specified
        self.start_epoch = 0
        self.start_step = 0
        state_dict = None
        self.checkpoint_path = os.path.join(save_dir, 'checkpoint.pt') if save_dir else None
        if ckpt is not None or (self.checkpoint_path is not None and os.path.exists(self.checkpoint_path)):
            # Load model parameters from existing checkpoint
            if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
                # loads from save dir even if args.ckpt is specified
                ckpt = self.checkpoint_path
                log(f'Found existing checkpoint {ckpt} in the experiment directory {save_dir}.')

                log(f'Loading checkpoint from {ckpt}.')
                if self.ddp:
                    dist.barrier()  # make sure that all processes load the model before optimizing it
                    map_location = {'cuda:%d' % 0: device}
                else:
                    map_location = self.rank
                state_dict = torch.load(ckpt, map_location=map_location, weights_only=False)
                model.load_state_dict(state_dict['state_dict'])
                self.start_epoch = state_dict['epoch']
                self.start_step = state_dict['step']
                log('Model with {} parameters loaded from epoch {}, step {}.'.format(capacity(model)[1],
                                                                                     self.start_epoch,
                                                                                     self.start_step))
            else:
                # Load checkpoint state dict into the existing model (GHN)
                log(f'Loading checkpoint from {ckpt}.')
                if self.ddp:
                    dist.barrier()  # make sure that all processes load the model before optimizing it
                    map_location = {'cuda:%d' % 0: device}
                else:
                    map_location = self.rank
                state_dict = torch.load(ckpt, map_location=map_location, weights_only=False)
                model.load_state_dict(state_dict['state_dict'])
                if 'epoch' in state_dict:
                    self.start_epoch = state_dict['epoch']
                if 'step' in state_dict:
                    self.start_step = state_dict['step']
                log('Model with {} parameters loaded from checkpoint.'.format(capacity(model)[1]))
                if 'epoch' in state_dict and 'step' in state_dict:
                    log('Resuming from epoch {}, step {}.'.format(self.start_epoch, self.start_step))

        self._is_ghn = isinstance(model, GHN) or (hasattr(model, 'module') and isinstance(model.module, GHN))
        if self.ddp:
            # Use find_unused_parameters=True for GHN because different models may use different decoder parts
            # This prevents "unused parameters" errors in DDP when some parameters don't receive gradients
            # Use static_graph=True because the GHN decoder structure is static across iterations,
            # even though the same decoder parameters are reused for multiple models in the same batch
            model = DDP(model, device_ids=[self.rank], output_device=self.rank, 
                       find_unused_parameters=True, static_graph=True)

        if compile_mode not in [None, 'none', False]:
            try:
                log(f'compiling the model using the {compile_mode} mode to improve efficiency (if pytorch>=2.0)...')
                model = torch.compile(model, mode=compile_mode)
                log('compiling the model succeeded!')
            except Exception as e:
                log('compiling the model failed: %s' % e)

        self._model = model

        self._reset(opt, opt_args, scheduler, scheduler_args, state_dict)

    def reset_metrics(self, epoch):
        self._step = 0
        if epoch > self.start_epoch:
            self.start_step = 0
        self.metrics = {'loss': AvgrageMeter(), 'top1': AvgrageMeter(), 'top5': AvgrageMeter()}
        if self.predparam_wd > 0:
            self.metrics['loss_predwd'] = AvgrageMeter()  # predicted parameter regularization loss
        self.logger = Logger(self.n_batches, start_step=self.start_step)

    def _reset(self, opt, opt_args, scheduler, scheduler_args, state_dict):

        assert 'lr' in opt_args, 'learning rate must be specified in opt_args'
        if opt.lower() == 'sgd':
            optimizer = torch.optim.SGD
        else:
            if opt.lower() == 'adam':
                optimizer = torch.optim.Adam
            elif opt.lower() == 'adamw':
                optimizer = torch.optim.AdamW
            elif opt.lower() == 'lamb':
                optimizer = Lamb
            else:
                raise NotImplementedError(opt)
            if 'momentum' in opt_args:
                del opt_args['momentum']

        self._optimizer = optimizer(self._model.parameters(), **opt_args)

        if scheduler.startswith('cosine-warmup'):

            def parse_arg(arg, default):
                p = scheduler.find(arg)
                if p > 0:
                    p_end = scheduler[p:].find('-')
                    return float(scheduler[p + len(arg):len(scheduler) if p_end == -1 else p + p_end])
                else:
                    return default

            warmup_steps = int(parse_arg('steps', 1000))  # number of warmup steps (default: 1000)
            cycles = 0.5
            warmup_lr = parse_arg('init_lr', 0.0001) / opt_args['lr']  # initial warmup lr (default: 0.0001)
            total_steps = self.epochs * self.n_batches  # total training steps

            def lr_lambda(step):
                # Based on https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
                if step < warmup_steps:
                    return warmup_lr + (1.0 - warmup_lr) * step / warmup_steps
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1. + math.cos(math.pi * cycles * 2.0 * progress)))
            self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        elif scheduler == 'cosine':
            self._scheduler = CosineAnnealingLR(self._optimizer, self.epochs)
        elif scheduler == 'step':
            self._scheduler = StepLR(self._optimizer, **scheduler_args)
        elif scheduler == 'mstep':
            self._scheduler = MultiStepLR(self._optimizer, **scheduler_args)
            # e.g. GHN-2 scheduler_args={'milestones'=[200, 250], 'gamma'=0.1}
        else:
            raise NotImplementedError(scheduler)

        if state_dict is not None and 'optimizer' in state_dict:
            if self.verbose:
                print('loading optimizer state')
            self._optimizer.load_state_dict(state_dict['optimizer'])

        # if training is resumed, adjust the learning rate
        if self.start_epoch > 0:
            self._scheduler.step(self.start_epoch)

        if self.amp:
            self.skipped_updates = 0

        self.reset_metrics(self.start_epoch)

        if state_dict is not None:
            if self.start_step >= self.n_batches - 1:
                self.start_step = 0
                self.start_epoch += 1  # resume from the next epoch
            else:
                self.start_step += 1  # resume from the next step

    def get_lr(self):
        for param_group in self._optimizer.param_groups:
            return param_group['lr']

    def scheduler_step(self):
        self._scheduler.step()

    def update(self, images, targets, graphs=None):

        def loss_check(loss_):
            if self.ddp:
                loss_avg_ = avg_ddp_metric(loss_)
                if torch.isnan(loss_avg_):
                    msg = f'rank {self.rank}, step {self._step}, the loss is {loss_}. ' \
                          f'Skip this batch, because the avg loss is {loss_avg_}.'
                    if self.verbose:
                        print(msg)
                    return msg
                else:
                    return loss_avg_
            elif torch.isnan(loss_):
                msg = f'the loss is {loss_}, unable to proceed. ' \
                      f'This issue may be fixed by restarting the script and loading the saved checkpoint ' \
                      f'using the --ckpt argument.'
                raise RuntimeError(msg)

            return

        logits = []
        loss = 0
        loss_predwd = None
        nan_loss = torch.tensor(torch.nan, device=self.rank)

        self._optimizer.zero_grad()
        if not self._model.training:
            self._model.train()

        try:
            with torch.cuda.amp.autocast(enabled=self.amp):

                if self._is_ghn:
                    # Predict parameters
                    if hasattr(graphs, 'nets') and len(graphs.nets) > 0:
                        models = graphs.nets
                    else:
                        # these are heavyweight Network objects that are less efficient but good for debugging
                        models = []
                        for nets_args in graphs.net_args:
                            # only for debugging (set is_imagenet_input and num_classes if needed)
                            models.append(Network(**nets_args))

                    models = self._model(models,
                                         graphs.to_device(self.device),
                                         bn_track_running_stats=True,
                                         keep_grads=True,
                                         reduce_graph=True
                                         )
                    if self.predparam_wd > 0:
                        total_norm = 0
                        for m in models:
                            for p in m.parameters():
                                total_norm += self.param_decay(p)

                        loss_predwd = self.predparam_wd * total_norm
                else:
                    models = self._model

                targets = targets.to(self.rank, non_blocking=True)  # loss will be computed on the main device
                targets_one_hot = targets
                images = images.to(self.rank, non_blocking=True)

                if self.mixup_fn is not None:
                    images, targets = self.mixup_fn(images, targets)

                if not isinstance(models, (list, tuple)):
                    models = [models]

                for model in models:
                    try:
                        out = model(images)
                    except:
                        print(model)
                        raise
                    y = out[0] if isinstance(out, tuple) else out
                    loss += self.criterion(y, targets)
                    if self.auxiliary:
                        loss += self.auxiliary_weight * self.criterion(out[1], targets)

                    logits.append(y.detach())

                # Concatenate logits across models
                logits = torch.stack(logits)  # num models x batch size x num classes

            if loss_predwd is not None:
                loss += loss_predwd

            loss = loss / len(logits)         # mean loss across models
            loss_avg = loss_check(loss)

            if self._step == 0 and self.ddp:
                if graphs is None:
                    net_idx = 0
                    n_graphs = 0
                else:
                    net_idx = graphs[0].net_idx
                    n_graphs = len(graphs)
                print(f'DDP: step {self._step}, rank {self.rank}, {n_graphs} graphs, '
                      f'net_idx {net_idx}, loss {loss}, loss_avg {loss_avg}, logits {logits.shape}')

            if isinstance(loss_avg, str):  # nan loss in any worker -> exit
                return loss_avg

            if self.amp:
                # Scales the loss, and calls backward()
                # to create scaled gradients
                self.scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self._optimizer)
            else:
                loss.backward()

            if self._step == 0 and self.rank == 0 and self.verbose:
                print_grads(self._model)

            if self.grad_clip > 0:
                parameters = []
                for group in self._optimizer.param_groups:
                    parameters.extend(group['params'])
                total_norm_clip = nn.utils.clip_grad_norm_(parameters, self.grad_clip)
            else:
                total_norm_clip = torch.zeros(1, device=self.rank)

            if self.amp:
                # Unscales gradients and calls
                # or skips optimizer.step()
                retval = self.scaler.step(self._optimizer)

                if retval is None and torch.logical_or(total_norm_clip.isnan(), total_norm_clip.isinf()):
                    self.skipped_updates += 1

                # Updates the scale for next iteration
                self.scaler.update()

                if self.amp_min_scale is not None:
                    # if the scale is too small then training is hindered, so we manually keep the scale large enough
                    scale = self.scaler._check_scale_growth_tracker('update')[0]
                    if scale < self.amp_min_scale:
                        self.scaler._scale = torch.tensor(self.amp_min_scale).to(scale)
            else:
                self._optimizer.step()

            targets = targets_one_hot.view(1, -1).expand(len(logits), -1).reshape(-1)
            logits = logits.reshape(-1, logits.shape[-1])

            # Update training metrics
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            n = len(targets)
            self.metrics['loss'].update((loss_avg if self.ddp else loss).item(), n)
            if loss_predwd is not None:
                self.metrics['loss_predwd'].update((avg_ddp_metric(loss_predwd) if self.ddp else loss_predwd).item(), n)
            self.metrics['top1'].update((avg_ddp_metric(prec1) if self.ddp else prec1).item(), n)
            self.metrics['top5'].update((avg_ddp_metric(prec5) if self.ddp else prec5).item(), n)

            self._step += 1

        except RuntimeError as err:

            print('error', 'rank ', self.rank, type(err), err, graphs.net_args if graphs is not None else '')
            loss = nan_loss

            print(traceback.format_exc())
            print(traceback.print_exc())
            if not self.ddp:
                raise

        loss_avg = loss_check(loss)
        if isinstance(loss_avg, str):  # oom in any worker -> exit
            raise RuntimeError(loss_avg)

        return self.metrics

    def update_lm(self, input_ids, labels, graphs=None, models=None):
        """
        Update method specifically for language model training.
        """
        import torch.nn.functional as F
        
        def loss_check(loss_):
            if self.ddp:
                loss_avg_ = avg_ddp_metric(loss_)
                if torch.isnan(loss_avg_):
                    msg = f'rank {self.rank}, step {self._step}, the loss is {loss_}. ' \
                          f'Skip this batch, because the avg loss is {loss_avg_}.'
                    if self.verbose:
                        print(msg)
                    return msg
                else:
                    return loss_avg_
            elif torch.isnan(loss_):
                msg = f'the loss is {loss_}, unable to proceed. ' \
                      f'This issue may be fixed by restarting the script and loading the saved checkpoint ' \
                      f'using the --ckpt argument.'
                raise RuntimeError(msg)

            return loss_

        logits = []
        loss = 0
        loss_predwd = None
        nan_loss = torch.tensor(torch.nan, device=self.rank)

        self._optimizer.zero_grad()
        if not self._model.training:
            self._model.train()

        try:
            with torch.cuda.amp.autocast(enabled=self.amp):

                if self._is_ghn:
                    # Predict parameters for language models
                    if models is not None:
                        predicted_models = self._model(models,
                                                     graphs.to_device(self.device),
                                                     bn_track_running_stats=True,
                                                     keep_grads=True,
                                                     reduce_graph=False
                                                     )
                    else:
                        # Fallback to original behavior
                        if hasattr(graphs, 'nets') and len(graphs.nets) > 0:
                            predicted_models = graphs.nets
                        else:
                            from .ops import Network
                            predicted_models = []
                            for nets_args in graphs.net_args:
                                predicted_models.append(Network(**nets_args))

                        predicted_models = self._model(predicted_models,
                                                     graphs.to_device(self.device),
                                                     bn_track_running_stats=True,
                                                     keep_grads=True,
                                                     reduce_graph=False
                                                     )
                    
                    if self.predparam_wd > 0:
                        total_norm = 0
                        for m in predicted_models:
                            for p in m.parameters():
                                total_norm += self.param_decay(p)
                        loss_predwd = self.predparam_wd * total_norm
                else:
                    predicted_models = self._model

                # Move data to device
                input_ids = input_ids.to(self.rank, non_blocking=True)
                labels = labels.to(self.rank, non_blocking=True)

                if not isinstance(predicted_models, (list, tuple)):
                    predicted_models = [predicted_models]

                # Compute language modeling loss for each predicted model
                for model in predicted_models:
                    try:
                        # Forward pass through the language model
                        output = model(input_ids)
                        # Extract logits from the output (model returns (logits, loss) tuple)
                        if isinstance(output, tuple):
                            logits_model = output[0]  # Extract logits
                        else:
                            logits_model = output
                        
                        # Language modeling loss (next token prediction)
                        model_loss = F.cross_entropy(logits_model.view(-1, logits_model.size(-1)), 
                                                   labels.view(-1), 
                                                   ignore_index=-100)
                        
                        # Check for NaN loss (edge case: all labels are -100)
                        if torch.isnan(model_loss):
                            print(f"Warning: NaN loss detected at step {self._step} for model {type(model).__name__}")
                            print(f"  Valid targets: {(labels != -100).sum().item()}")
                            print(f"  Total targets: {labels.numel()}")
                            # Skip this model's loss contribution
                            continue
                        
                        loss += model_loss
                        logits.append(logits_model.detach())
                        # Clear intermediate tensors to save memory
                        del logits_model
                    except Exception as e:
                        print(f"Error in model forward pass: {e}")
                        print(f"Model: {model}")
                        raise

                # Average loss across models
                if len(logits) > 0:  # Only if we have valid models
                    loss = loss / len(logits)
                else:
                    # All models produced NaN loss - skip this batch
                    print(f"Warning: All models produced NaN loss at step {self._step}")
                    print(f"  Skipping this batch")
                    return None
                
                # Note: Don't delete predicted_models here - they're needed for backward pass
                # They will be cleaned up by Python's GC after backward pass

            if loss_predwd is not None:
                loss += loss_predwd

            loss_avg = loss_check(loss)

            if self._step == 0 and self.ddp:
                if graphs is None:
                    net_idx = 0
                    n_graphs = 0
                else:
                    net_idx = getattr(graphs, 'net_idx', 0)
                    n_graphs = len(graphs) if hasattr(graphs, '__len__') else 1
                print(f'DDP: step {self._step}, rank {self.rank}, {n_graphs} graphs, '
                      f'net_idx {net_idx}, loss {loss}, loss_avg {loss_avg}')

            if isinstance(loss_avg, str):  # nan loss in any worker -> exit
                return loss_avg

            # Stack logits before backward to save memory (keep list structure for metrics)
            if len(logits) > 0 and isinstance(logits[0], torch.Tensor):
                logits_stack = torch.stack(logits)
                del logits
                logits = [logits_stack]  # Keep for metrics calculation
            else:
                logits = []

            if self.amp:
                # Scales the loss, and calls backward()
                # to create scaled gradients
                self.scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self._optimizer)
            else:
                loss.backward()
            
            # Clear memory after backward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self._step == 0 and self.rank == 0 and self.verbose:
                print_grads(self._model)

            if self.grad_clip > 0:
                parameters = []
                for group in self._optimizer.param_groups:
                    parameters.extend(group['params'])
                total_norm_clip = nn.utils.clip_grad_norm_(parameters, self.grad_clip)
            else:
                total_norm_clip = torch.zeros(1, device=self.rank)

            if self.amp:
                # Unscales gradients and calls
                # or skips optimizer.step()
                retval = self.scaler.step(self._optimizer)

                if retval is None and torch.logical_or(total_norm_clip.isnan(), total_norm_clip.isinf()):
                    self.skipped_updates += 1

                # Updates the scale for next iteration
                self.scaler.update()

                if self.amp_min_scale is not None:
                    # if the scale is too small then training is hindered, so we manually keep the scale large enough
                    scale = self.scaler._check_scale_growth_tracker('update')[0]
                    if scale < self.amp_min_scale:
                        self.scaler._scale = torch.tensor(self.amp_min_scale).to(scale)
            else:
                self._optimizer.step()

            # Update training metrics (for language models, we use perplexity instead of accuracy)
            if len(logits) > 0:
                # Calculate perplexity as a metric with overflow protection
                # logits is already a list with one stacked tensor from memory optimization above
                if isinstance(logits[0], torch.Tensor) and logits[0].dim() > 2:
                    avg_logits = logits[0].mean(0)  # Average logits across models
                else:
                    avg_logits = torch.stack(logits).mean(0)  # Fallback if structure is different
                cross_entropy_loss = F.cross_entropy(avg_logits.view(-1, avg_logits.size(-1)), 
                                                    labels.view(-1), 
                                                    ignore_index=-100)
                # Clamp loss to prevent overflow in perplexity calculation
                clamped_loss = torch.clamp(cross_entropy_loss, max=10.0)  # exp(10) ≈ 22,026
                perplexity = torch.exp(clamped_loss)
                
                n = len(labels.view(-1))
                self.metrics['loss'].update((loss_avg if self.ddp else loss).item(), n)
                if loss_predwd is not None:
                    self.metrics['loss_predwd'].update((avg_ddp_metric(loss_predwd) if self.ddp else loss_predwd).item(), n)
                # Store perplexity in top1 metric for language models
                self.metrics['top1'].update((avg_ddp_metric(perplexity) if self.ddp else perplexity).item(), n)
                # Keep top5 for compatibility (set to same as top1)
                self.metrics['top5'].update((avg_ddp_metric(perplexity) if self.ddp else perplexity).item(), n)
                
                # Clear logits after metrics calculation
                del avg_logits, cross_entropy_loss, clamped_loss, perplexity

            self._step += 1

        except RuntimeError as err:
            # Check if it's an OOM error
            if 'out of memory' in str(err).lower() or 'cuda' in str(err).lower():
                print(f'OOM error on rank {self.rank}, step {self._step}')
                # Clear cache to potentially recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print('error', 'rank ', self.rank, type(err), err, getattr(graphs, 'net_args', '') if graphs is not None else '')
            loss = nan_loss

            import traceback
            print(traceback.format_exc())
            print(traceback.print_exc())
            if not self.ddp:
                raise

        loss_avg = loss_check(loss)
        if isinstance(loss_avg, str):  # oom in any worker -> exit
            raise RuntimeError(loss_avg)

        return self.metrics

    def save(self, epoch, step, config, save_freq=300, interm_epoch=5):

        # save every save_freq steps, so that training can be resumed if one epoch takes a long time
        if not ((((step + 1) % save_freq == 0) or step == self.n_batches - 1) and self.rank == 0):
            return

        state_dict = {'state_dict': (self._model.module if hasattr(self._model, 'module')
                                     else self._model).state_dict(),
                      'optimizer': self._optimizer.state_dict(),
                      'epoch': epoch,
                      'step': step
                      }
        state_dict.update(config)
        torch.save(state_dict, self.checkpoint_path)
        log('\nsaved the checkpoint to {} at epoch={}, step={}'.format(self.checkpoint_path, epoch, step))

        if (epoch + 1) % interm_epoch == 0 or epoch == 0:
            checkpoint_path_interm = self.checkpoint_path.replace('.pt', '_epoch%d.pt' % (epoch + 1))
            torch.save(state_dict, checkpoint_path_interm)
            log('saved the intermediate checkpoint to {}'.format(checkpoint_path_interm))

    def log(self, step=None):
        step_ = self._step if step is None else (step + 1)
        if step_ % self.log_interval == 0 or step_ >= self.n_batches - 1 or step_ == 1:
            metrics = {metric: value.avg for metric, value in self.metrics.items()}
            if self.amp:
                metrics['amp_scale'] = self.scaler._check_scale_growth_tracker('update')[0].item()
            self.logger(step_, metrics)
