#!/usr/bin/env python3
"""
Simple test script for GHN-3 training with a single language model.
This script tests the core training loop without the complexity of multiple models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import warnings
import sys
import os
from tqdm import tqdm

# Add the project root to the path
sys.path.append('/Users/anshulkiyawat/Projects/Capstone')

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import our modules
from lmghn3.CustomGHN3.nn import GHN3
from lmghn3.models.mini_gpt import GPTDecoderLM, GPTConfig
from lmghn3.Dataloader.wikitext2_loader import build_wikitext2
from lmghn3.CustomGHN3.graph import Graph
from lmghn3.CustomGHN3.trainer import Trainer

def create_simple_model(vocab_size=1000, seq_len=64, max_len=128):
    """Create a simple transformer model for testing."""
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=max_len,
        d_model=128,
        n_head=2,
        n_layer=2,
        d_ff=256,
        p_drop=0.1
    )
    return GPTDecoderLM(config)

def create_simple_data(vocab_size=1000, seq_len=64, batch_size=4, num_batches=10):
    """Create simple synthetic data for testing."""
    data = []
    for _ in range(num_batches):
        # Create random input sequences
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Labels are the same as input_ids (next token prediction)
        labels = input_ids.clone()
        data.append((input_ids, labels))
    return data

def test_ghn_training():
    """Test GHN-3 training with a single model."""
    print("=== Testing GHN-3 Training with Single Model ===")
    
    # Configuration
    vocab_size = 1000
    seq_len = 64
    max_len = 128
    batch_size = 4
    num_batches = 5
    
    # Create a simple target model
    print("Creating target model...")
    target_model = create_simple_model(vocab_size, seq_len, max_len)
    print(f"Target model parameters: {sum(p.numel() for p in target_model.parameters())}")
    
    # Create GHN-3 model
    print("Creating GHN-3 model...")
    ghn_config = {
        'max_shape': (seq_len, seq_len, 8, 8),
        'num_classes': vocab_size,
        'hid': 64,
        'layers': 2,
        'heads': 2,
        'is_ghn2': False
    }
    ghn = GHN3(**ghn_config, debug_level=1)
    print(f"GHN-3 parameters: {sum(p.numel() for p in ghn.parameters())}")
    
    # Create graph for the target model
    print("Creating computational graph...")
    graph = Graph(target_model, dense=True, verbose=False)
    print(f"Graph nodes: {len(graph.node_info)}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=ghn,
        opt='adam',
        opt_args={'lr': 0.001, 'weight_decay': 1e-5},
        scheduler='step',
        scheduler_args={'step_size': 10, 'gamma': 0.1},
        n_batches=num_batches,
        grad_clip=5.0,
        device='cpu',
        amp=False
    )
    
    # Create simple data
    print("Creating test data...")
    test_data = create_simple_data(vocab_size, seq_len, batch_size, num_batches)
    
    # Test training loop
    print("\n=== Starting Training Loop ===")
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        
        # Create progress bar for batches
        pbar = tqdm(enumerate(test_data), total=len(test_data), 
                   desc=f'Epoch {epoch+1}/2', unit='batch')
        
        for batch_idx, (input_ids, labels) in pbar:
            try:
                # Test parameter prediction
                print("    Predicting parameters...")
                predicted_model = ghn(target_model, graphs=[graph], keep_grads=True)
                
                # Forward pass
                print("    Forward pass...")
                outputs = predicted_model(input_ids)
                loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1))
                perplexity = torch.exp(loss)
                
                print(f"    Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.4f}")
                
                # Test trainer update
                print("    Testing trainer update...")
                from lmghn3.CustomGHN3.graph import GraphBatch
                graph_batch = GraphBatch([graph], dense=True)
                trainer.update_lm(input_ids, labels, graphs=graph_batch, models=[predicted_model])
                
                # Update progress bar with metrics
                pbar.set_postfix({
                    'loss': f'{trainer.metrics["loss"].avg:.4f}',
                    'perplexity': f'{trainer.metrics["top1"].avg:.2f}'
                })
                
            except Exception as e:
                print(f"    ERROR in batch {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                pbar.close()
                return False
        
        pbar.close()
    
    print("\n=== Training Test Completed Successfully! ===")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GHN-3 training with single model')
    parser.add_argument('--vocab_size', type=int, default=1000, help='vocabulary size')
    parser.add_argument('--seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_batches', type=int, default=5, help='number of batches to test')
    
    args = parser.parse_args()
    
    success = test_ghn_training()
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)
