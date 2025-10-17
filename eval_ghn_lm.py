#!/usr/bin/env python3
"""
Evaluates a trained GHN on language model architectures.

This script is based on the existing eval_ghn.py framework but adapted for language models.
It evaluates how well the GHN can predict parameters for various language model architectures.

Example:
    # Evaluating on a subset of language models:
    python eval_ghn_lm.py --ckpt Experiment/20917896/best_model.pt --num_models 10
    
    # Evaluating on all available language models:
    python eval_ghn_lm.py --ckpt Experiment/20917896/best_model.pt --num_models -1
"""

import torch
import time
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np

# Import GHN components
from lmghn3.CustomGHN3 import GHN3, Graph, GraphBatch
from lmghn3.Dataloader.lm_arch_loader import build_ghn_variants_dataloader, create_model_from_config
from lmghn3.Dataloader.wikitext2_loader import build_wikitext2


class AvgrageMeter:
    """Average meter for tracking metrics."""
    def __init__(self, dispersion='std'):
        self.reset()
        self.dispersion = dispersion

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.dispersion = 0
        self.values = []

    def update(self, val, n=1):
        self.values.append(val)
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        if self.dispersion == 'std' and len(self.values) > 1:
            self.dispersion = np.std(self.values)
        elif self.dispersion == 'max':
            self.dispersion = max(self.values) - min(self.values)


def load_trained_ghn(model_path: str, config_path: str, device: str = None) -> Tuple[GHN3, Dict]:
    """Load a trained GHN model and its configuration."""
    print(f"Loading trained GHN from: {model_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Extract GHN config
    ghn_config = config_data['config']
    print(f"GHN Config: {ghn_config}")
    
    # Create GHN model
    ghn = GHN3(
        max_shape=ghn_config['max_shape'],
        num_classes=ghn_config['num_classes'],
        hypernet=ghn_config['hypernet'],
        decoder=ghn_config['decoder'],
        weight_norm=ghn_config['weight_norm'],
        ve=ghn_config['ve'],
        layernorm=ghn_config['layernorm'],
        hid=ghn_config['hid'],
        layers=ghn_config['layers'],
        heads=ghn_config['heads'],
        is_ghn2=ghn_config['is_ghn2'],
        exclude_embeddings=ghn_config['exclude_embeddings']
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # Full checkpoint with optimizer state, etc.
        state_dict = checkpoint['state_dict']
        print(f"Loaded full checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Direct state dict
        state_dict = checkpoint
    
    ghn.load_state_dict(state_dict)
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ghn.to(device)
    ghn.eval()
    
    print(f"Successfully loaded GHN with {sum(p.numel() for p in ghn.parameters())} parameters")
    return ghn, config_data


def evaluate_model_on_wikitext2(model, test_loader, device='cuda', max_samples=1000):
    """Evaluate a language model on WikiText-2 test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if total_tokens >= max_samples:
                break
                
            # Handle the new batch format from wikitext2_loader
            if isinstance(batch, dict):
                inputs = batch['input_ids']
                targets = batch['labels']
            else:
                inputs, targets = batch
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_loss += loss.item() * inputs.numel()
            total_tokens += inputs.numel()
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity, total_tokens
    else:
        return float('inf'), 0


def main():
    parser = argparse.ArgumentParser(description='Evaluation of GHN on Language Models')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to trained GHN checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration JSON file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run evaluation on (auto-detect if not specified)')
    parser.add_argument('--num_models', type=int, default=50,
                       help='Number of models to evaluate (-1 for all)')
    parser.add_argument('--max_wikitext_samples', type=int, default=1000,
                       help='Maximum number of samples to evaluate on WikiText-2')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print("=" * 60)
    print("GHN-3 Language Model Evaluation")
    print("=" * 60)
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {args.device}")
    
    # Load trained GHN
    ghn, config_data = load_trained_ghn(args.ckpt, args.config, args.device)
    
    # Load WikiText-2 test data
    seq_len = config_data.get('seq_len', 64)
    print(f"Loading WikiText-2 test data (seq_len={seq_len})...")
    wikitext_data = build_wikitext2(
        tokenizer_name="gpt2",
        seq_len=seq_len,
        batch_size=32,
        num_workers=0
    )
    test_loader = wikitext_data['val_loader']  # Use validation set as test
    
    # Create evaluation dataloader
    print("Creating evaluation dataloader...")
    eval_dataloader, all_configs = build_ghn_variants_dataloader(
        batch_size=1,  # Evaluate one model at a time
        vocab_size=50257,
        max_len=seq_len,
        device=args.device,
        num_workers=0,
        shuffle=False  # Deterministic evaluation
    )
    
    # Select models to evaluate
    if args.num_models == -1:
        models_to_eval = all_configs
    else:
        # Select a diverse subset
        step = max(1, len(all_configs) // args.num_models)
        models_to_eval = all_configs[::step][:args.num_models]
    
    print(f"Evaluating {len(models_to_eval)} language models...")
    
    # Evaluation metrics
    perplexities = AvgrageMeter('std')
    param_counts = []
    param_norms = AvgrageMeter('std')
    evaluation_times = []
    
    start_all = time.time()
    
    for m_ind, config in enumerate(models_to_eval):
        try:
            model_type = config['model_type']
            model_name = config['name']
            model_config = config.copy()
            model_config.pop('name', None)
            model_config.pop('model_type', None)
            
            print(f"\n{m_ind + 1}/{len(models_to_eval)}: {model_name}")
            
            # Create target model
            target_model = create_model_from_config(model_type, model_config, args.device)
            n_params = sum(p.numel() for p in target_model.parameters()) / 1e6
            param_counts.append(n_params)
            
            if args.verbose:
                print(f"  Model type: {model_type}")
                print(f"  Parameters: {n_params:.2f}M")
            
            # Create graph for GHN
            graph = Graph(target_model, ve_cutoff=50, dense=True, verbose=False, 
                         exclude_embeddings=config_data['config']['exclude_embeddings'])
            graph_batch = GraphBatch([graph], dense=True)
            graph_batch.to_device(args.device)
            
            # Predict parameters using GHN
            if args.device != 'cpu':
                torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                predicted_model = ghn(target_model, graphs=graph_batch, bn_track_running_stats=True, reduce_graph=True)
            
            eval_time = time.time() - start
            evaluation_times.append(eval_time)
            
            # Calculate parameter norm
            total_norm = torch.norm(torch.stack([p.norm() for p in predicted_model.parameters()]), 2)
            param_norms.update(total_norm.item())
            
            if args.verbose:
                print(f"  Parameter norm: {total_norm.item():.2f}")
                print(f"  Prediction time: {eval_time:.2f}s")
            
            # Evaluate on WikiText-2
            print(f"  Evaluating on WikiText-2...")
            perplexity, tokens_evaluated = evaluate_model_on_wikitext2(
                predicted_model, test_loader, args.device, args.max_wikitext_samples
            )
            
            if perplexity != float('inf'):
                perplexities.update(perplexity)
                print(f"  Perplexity: {perplexity:.2f} (evaluated on {tokens_evaluated} tokens)")
            else:
                print(f"  Failed to evaluate on WikiText-2")
            
        except Exception as e:
            print(f"ERROR for model {model_name}: {e}")
            continue
    
    total_time = time.time() - start_all
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Models evaluated: {len(models_to_eval)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per model: {np.mean(evaluation_times):.2f} seconds")
    print(f"Average parameters: {np.mean(param_counts):.2f}M ± {np.std(param_counts):.2f}M")
    print(f"Average parameter norm: {param_norms.avg:.2f} ± {param_norms.dispersion:.2f}")
    
    if perplexities.cnt > 0:
        print(f"Average perplexity: {perplexities.avg:.2f} ± {perplexities.dispersion:.2f}")
        print(f"Perplexity range: [{min(perplexities.values):.2f}, {max(perplexities.values):.2f}]")
    else:
        print("No models successfully evaluated on WikiText-2")
    
    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': args.ckpt,
        'config': args.config,
        'models_evaluated': len(models_to_eval),
        'total_time': total_time,
        'avg_time_per_model': np.mean(evaluation_times),
        'avg_parameters': float(np.mean(param_counts)),
        'std_parameters': float(np.std(param_counts)),
        'avg_param_norm': param_norms.avg,
        'std_param_norm': param_norms.dispersion,
        'avg_perplexity': perplexities.avg if perplexities.cnt > 0 else None,
        'std_perplexity': perplexities.dispersion if perplexities.cnt > 0 else None,
        'perplexity_range': [min(perplexities.values), max(perplexities.values)] if perplexities.cnt > 0 else None,
        'successful_evaluations': perplexities.cnt
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
    
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
