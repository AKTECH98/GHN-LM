"""
Language Model Trainer
Handles training and validation of language models with support for both config files and command-line arguments
"""

import os
import time
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from simple_logger import SimpleLogger


class Trainer:
    """Trainer class for language models."""
    
    def __init__(self, model, train_loader, val_loader, device, args=None, training_config=None, model_config=None, data_config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        self.training_config = training_config
        self.model_config = model_config
        self.data_config = data_config
        
        # Get job ID from environment or generate one (same as GHN training)
        self.job_id = os.environ.get('SLURM_JOB_ID', f'lm_{int(time.time())}')
        
        # Create directory structure (same as GHN training)
        self.logging_dir = 'logging'
        self.experiment_dir = 'Experiment'
        self.job_experiment_dir = os.path.join(self.experiment_dir, self.job_id)
        
        # Create directories
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.job_experiment_dir, exist_ok=True)
        
        # Set random seeds for reproducibility (if seed is provided)
        self.seed = getattr(training_config, 'seed', None) if training_config else getattr(args, 'seed', None)
        if self.seed is not None:
            self._set_seed(self.seed)
            print(f"🌱 Random seed set to: {self.seed}")
        
        # Use training_config if available, otherwise fall back to args
        if training_config:
            lr = training_config.learning_rate
            weight_decay = training_config.weight_decay
        else:
            lr = args.learning_rate
            weight_decay = args.weight_decay
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        if training_config:
            epochs = training_config.epochs
            base_lr = training_config.learning_rate
        else:
            epochs = args.epochs
            base_lr = args.learning_rate
            
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=base_lr * 0.1
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Setup TensorBoard logging (same as GHN training)
        model_name = model_config.model_type if model_config else (args.model if args else "unknown")
        tensorboard_log_dir = os.path.join(self.logging_dir, self.job_id)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f'TensorBoard logs will be saved to: {tensorboard_log_dir}')
        print(f'Experiment data will be saved to: {self.job_experiment_dir}')
        
        # Save config metadata to experiment directory (same structure as GHN training)
        if training_config and model_config and data_config:
            # Create config in the same structure as input YAML files
            experiment_config = {
                "job_id": self.job_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": {
                    "model_type": model_config.model_type,
                    "vocab_size": "TBD",  # Will be updated after data loading
                    "d_model": model_config.d_model,
                    "n_layer": model_config.n_layer,
                    "n_head": model_config.n_head,
                    "d_ff": model_config.d_ff,
                    "max_seq_len": model_config.max_seq_len,
                    "p_drop": model_config.p_drop
                },
                "training": {
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                    "weight_decay": training_config.weight_decay,
                    "warmup_steps": training_config.warmup_steps,
                    "max_grad_norm": training_config.max_grad_norm,
                    "save_interval": training_config.save_interval,
                    "eval_interval": training_config.eval_interval,
                    "log_interval": training_config.log_interval,
                    "device": training_config.device,
                    "mixed_precision": training_config.mixed_precision,
                    "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                    "seed": training_config.seed
                },
                "data": {
                    "seq_len": data_config.seq_len,
                    "num_workers": data_config.num_workers
                }
            }
            
            # Save config to experiment directory
            config_path = os.path.join(self.job_experiment_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(experiment_config, f, indent=2)
            print(f"📋 Config saved to: {config_path}")
        
        # Setup simple logger (will be updated with actual vocab_size after data loading)
        if training_config and model_config and data_config:
            config = {
                "d_model": model_config.d_model,
                "n_layer": model_config.n_layer,
                "seq_len": data_config.seq_len,
                "batch_size": training_config.batch_size,
                "vocab_size": "TBD",  # Will be updated after data loading
                "dropout": model_config.p_drop
            }
            self.logger = SimpleLogger(model_config.model_type, config, training_config, self.job_id)
        else:
            config = {
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "vocab_size": "TBD",  # Will be updated after data loading
                "dropout": getattr(args, 'dropout', 0.1)
            }
            self.logger = SimpleLogger(args.model, config, args, self.job_id)
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def update_logger_config(self, vocab_size):
        """Update logger config with actual vocab size from tokenizer."""
        self.logger.log_data["config"]["vocab_size"] = vocab_size
        
        # Also update the experiment config.json with actual vocab size
        if hasattr(self, 'job_experiment_dir'):
            config_path = os.path.join(self.job_experiment_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                config_data["model"]["vocab_size"] = vocab_size
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                print(f"📋 Updated config with vocab_size: {vocab_size}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Get epochs from config or args
        epochs = self.training_config.epochs if self.training_config else (self.args.epochs if self.args else 10)
        log_interval = self.training_config.log_interval if self.training_config else (self.args.log_interval if self.args else 10)
        grad_clip = self.training_config.max_grad_norm if self.training_config else (self.args.grad_clip if self.args else 1.0)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
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
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
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
            if batch_idx % log_interval == 0:
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
            'args': self.args,
            'training_config': self.training_config
        }
        
        # Save checkpoint every 5 epochs and at epoch 2 (same as GHN training)
        if (epoch + 1) % 5 == 0 or epoch + 1 == 2:
            checkpoint_path = os.path.join(self.job_experiment_dir, f'epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved to {checkpoint_path}")
        
        # Save best model (same as GHN training)
        if is_best:
            best_path = os.path.join(self.job_experiment_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved to {best_path}")
    
    def train(self):
        """Main training loop."""
        # Get parameters from config or args
        if self.training_config:
            model_name = self.model_config.model_type
            epochs = self.training_config.epochs
            batch_size = self.training_config.batch_size
            learning_rate = self.training_config.learning_rate
        else:
            model_name = self.args.model if self.args else "unknown"
            epochs = self.args.epochs if self.args else 10
            batch_size = self.args.batch_size if self.args else 32
            learning_rate = self.args.learning_rate if self.args else 0.001
        
        print(f"🚀 Starting training of {model_name} model")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
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
        print(f"   TensorBoard logs saved to: {os.path.join(self.logging_dir, self.job_id)}")
        print(f"   Checkpoints saved to: {self.job_experiment_dir}")
        print(f"\n📊 TensorBoard Metrics Available:")
        print(f"   - Train/Loss: Batch-level training losses")
        print(f"   - Val/Loss: Epoch-level validation losses")
        print(f"   - Epoch/Train_Loss: Epoch-level training losses")
        print(f"   - Epoch/Val_Loss: Epoch-level validation losses")
        print(f"   - Epoch/Train_Perplexity: Training perplexity")
        print(f"   - Epoch/Val_Perplexity: Validation perplexity")
        
        # Close tensorboard writer
        self.writer.close()
