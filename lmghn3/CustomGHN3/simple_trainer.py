"""
Simplified trainer for GHN training without DDP support.
Based on the original trainer but simplified for single-GPU training.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, LambdaLR
from .utils import log


class SimpleTrainer:
    """Simplified trainer for GHN without DDP."""
    
    def __init__(self,
                 model,
                 opt,
                 opt_args,
                 scheduler,
                 n_batches,
                 grad_clip=5,
                 device='cuda',
                 log_interval=100,
                 label_smoothing=0,
                 predparam_wd=0,
                 scheduler_args=None,
                 save_dir=None,
                 ckpt=None,
                 epochs=None,
                 verbose=False,
                 amp=False,
                 amp_min_scale=None,
                 amp_growth_interval=2000,
                 compile_mode=None,
                 beta=1e-5):
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.n_batches = n_batches
        self.grad_clip = grad_clip
        self.device = device
        self.log_interval = log_interval
        self.amp = amp
        self.amp_min_scale = amp_min_scale
        self.predparam_wd = predparam_wd
        self.epochs = epochs
        self.verbose = verbose
        
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler(growth_interval=amp_growth_interval)
        
        if predparam_wd > 0:
            self.param_decay = lambda p: torch.norm(p, p='fro')
        
        model.to(self.device)
        
        # Handle checkpoint loading
        self.start_epoch = 0
        self.start_step = 0
        state_dict = None
        self.checkpoint_path = os.path.join(save_dir, 'checkpoint.pt') if save_dir else None
        
        if ckpt is not None or (self.checkpoint_path is not None and os.path.exists(self.checkpoint_path)):
            if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
                ckpt = self.checkpoint_path
                log(f'Found existing checkpoint {ckpt}')
                log(f'Loading checkpoint from {ckpt}')
                state_dict = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(state_dict['state_dict'])
                self.start_epoch = state_dict['epoch']
                self.start_step = state_dict['step']
                log(f'Model loaded from epoch {self.start_epoch}, step {self.start_step}')
        
        # Compile model if requested
        if compile_mode not in [None, 'none', False]:
            try:
                log(f'Compiling model using {compile_mode} mode...')
                model = torch.compile(model, mode=compile_mode)
                log('Model compilation succeeded!')
            except Exception as e:
                log(f'Model compilation failed: {e}')
        
        self._model = model
        self._reset(opt, opt_args, scheduler, scheduler_args, state_dict)
        
    def _reset(self, opt, opt_args, scheduler, scheduler_args, state_dict):
        """Reset optimizer and scheduler."""
        assert 'lr' in opt_args, 'learning rate must be specified in opt_args'
        
        if opt.lower() == 'sgd':
            optimizer = torch.optim.SGD
        elif opt.lower() == 'adam':
            optimizer = torch.optim.Adam
        elif opt.lower() == 'adamw':
            optimizer = torch.optim.AdamW
        else:
            raise NotImplementedError(opt)
        
        if opt.lower() != 'sgd' and 'momentum' in opt_args:
            del opt_args['momentum']
        
        self._optimizer = optimizer(self._model.parameters(), **opt_args)
        
        # Setup scheduler
        if scheduler.startswith('cosine-warmup'):
            def parse_arg(arg, default):
                p = scheduler.find(arg)
                if p > 0:
                    p_end = scheduler[p:].find('-')
                    return float(scheduler[p + len(arg):len(scheduler) if p_end == -1 else p + p_end])
                else:
                    return default
            
            warmup_steps = int(parse_arg('steps', 5))
            cycles = 0.5
            warmup_lr = parse_arg('init_lr', 1e-5) / opt_args['lr']
            
            def lr_lambda(step):
                if step < warmup_steps - 1:
                    return np.linspace(warmup_lr, 1, warmup_steps)[step]
                progress = float(step - warmup_steps) / float(max(1, self.epochs - warmup_steps))
                return max(0.0, 0.5 * (1. + math.cos(math.pi * cycles * 2.0 * progress)))
            
            self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)
        elif scheduler == 'cosine':
            self._scheduler = CosineAnnealingLR(self._optimizer, self.epochs)
        elif scheduler == 'step':
            self._scheduler = StepLR(self._optimizer, **scheduler_args)
        elif scheduler == 'mstep':
            self._scheduler = MultiStepLR(self._optimizer, **scheduler_args)
        else:
            raise NotImplementedError(scheduler)
        
        if state_dict is not None and 'optimizer' in state_dict:
            if self.verbose:
                log('Loading optimizer state')
            self._optimizer.load_state_dict(state_dict['optimizer'])
        
        if self.start_epoch > 0:
            self._scheduler.step(self.start_epoch)
        
        if self.amp:
            self.skipped_updates = 0
        
        self.reset_metrics(self.start_epoch)
        
        if state_dict is not None:
            if self.start_step >= self.n_batches - 1:
                self.start_step = 0
                self.start_epoch += 1
            else:
                self.start_step += 1
    
    def reset_metrics(self, epoch):
        """Reset training metrics."""
        self._step = 0
        if epoch > self.start_epoch:
            self.start_step = 0
        
        self.metrics = {
            'loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter()
        }
        if self.predparam_wd > 0:
            self.metrics['loss_predwd'] = AverageMeter()
    
    def get_lr(self):
        """Get current learning rate."""
        for param_group in self._optimizer.param_groups:
            return param_group['lr']
    
    def scheduler_step(self):
        """Step the learning rate scheduler."""
        self._scheduler.step()
    
    def update(self, input_ids, targets, graphs=None):
        """Update model with one batch."""
        self._optimizer.zero_grad()
        if not self._model.training:
            self._model.train()
        
        try:
            with torch.cuda.amp.autocast(enabled=self.amp):
                # Predict parameters using GHN
                if graphs is not None and hasattr(graphs, 'nets'):
                    models = self._model(graphs.nets,
                                       graphs.to_device(self.device),
                                       bn_track_running_stats=True,
                                       keep_grads=True,
                                       reduce_graph=True)
                    
                    if self.predparam_wd > 0:
                        total_norm = 0
                        for m in models:
                            for p in m.parameters():
                                total_norm += self.param_decay(p)
                        loss_predwd = self.predparam_wd * total_norm
                    else:
                        loss_predwd = None
                else:
                    models = [self._model]
                    loss_predwd = None
                
                # Move targets to device
                targets = targets.to(self.device, non_blocking=True)
                input_ids = input_ids.to(self.device, non_blocking=True)
                
                # Forward pass through models
                logits = []
                loss = 0
                
                for model in models:
                    try:
                        out = model(input_ids)
                        y = out[0] if isinstance(out, tuple) else out
                        loss += self.criterion(y, targets)
                        logits.append(y.detach())
                    except Exception as e:
                        log(f'Error in model forward pass: {e}')
                        raise
                
                # Average loss across models
                loss = loss / len(logits)
                
                # Add predicted parameter regularization
                if loss_predwd is not None:
                    loss += loss_predwd
                
                # Check for NaN loss
                if torch.isnan(loss):
                    log(f'NaN loss detected at step {self._step}')
                    return None
            
            # Backward pass
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self._optimizer)
            else:
                loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                total_norm = nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
            else:
                total_norm = torch.zeros(1, device=self.device)
            
            # Optimizer step
            if self.amp:
                retval = self.scaler.step(self._optimizer)
                if retval is None and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
                    self.skipped_updates += 1
                self.scaler.update()
                
                if self.amp_min_scale is not None:
                    scale = self.scaler._check_scale_growth_tracker('update')[0]
                    if scale < self.amp_min_scale:
                        self.scaler._scale = torch.tensor(self.amp_min_scale).to(scale)
            else:
                self._optimizer.step()
            
            # Update metrics
            targets_expanded = targets.view(1, -1).expand(len(logits), -1).reshape(-1)
            logits_flat = torch.stack(logits).reshape(-1, logits[0].shape[-1])
            
            prec1, prec5 = accuracy(logits_flat, targets_expanded, topk=(1, 5))
            n = len(targets_expanded)
            
            self.metrics['loss'].update(loss.item(), n)
            if loss_predwd is not None:
                self.metrics['loss_predwd'].update(loss_predwd.item(), n)
            self.metrics['top1'].update(prec1.item(), n)
            self.metrics['top5'].update(prec5.item(), n)
            
            self._step += 1
            
        except Exception as e:
            log(f'Error in update: {e}')
            raise
        
        return self.metrics
    
    def log(self, step=None):
        """Log training metrics."""
        step_ = self._step if step is None else (step + 1)
        if step_ % self.log_interval == 0 or step_ >= self.n_batches - 1 or step_ == 1:
            metrics = {metric: value.avg for metric, value in self.metrics.items()}
            if self.amp:
                metrics['amp_scale'] = self.scaler._check_scale_growth_tracker('update')[0].item()
            
            log_str = f'Step {step_}/{self.n_batches}'
            for name, value in metrics.items():
                if name == 'loss':
                    log_str += f', Loss: {value:.4f}'
                elif name == 'top1':
                    log_str += f', Top1: {value:.2f}%'
                elif name == 'top5':
                    log_str += f', Top5: {value:.2f}%'
                elif name == 'amp_scale':
                    log_str += f', Scale: {value:.0f}'
            
            log(log_str)
    
    def save(self, epoch, step, config, save_freq=300, interm_epoch=5):
        """Save checkpoint."""
        if not ((((step + 1) % save_freq == 0) or step == self.n_batches - 1)):
            return
        
        state_dict = {
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'epoch': epoch,
            'step': step
        }
        state_dict.update(config)
        
        torch.save(state_dict, self.checkpoint_path)
        log(f'Saved checkpoint to {self.checkpoint_path} at epoch={epoch}, step={step}')
        
        if (epoch + 1) % interm_epoch == 0 or epoch == 0:
            checkpoint_path_interm = self.checkpoint_path.replace('.pt', f'_epoch{epoch+1}.pt')
            torch.save(state_dict, checkpoint_path_interm)
            log(f'Saved intermediate checkpoint to {checkpoint_path_interm}')


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
