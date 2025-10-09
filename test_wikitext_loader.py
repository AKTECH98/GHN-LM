#!/usr/bin/env python3
"""
Test script for WikiText-2 dataloader functionality.

This script tests the wikitext2_loader module to ensure it's working correctly.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# Add the lmghn3 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lmghn3'))

def test_wikitext_loader():
    """Test the WikiText-2 dataloader functionality."""
    print("Testing WikiText-2 Dataloader...")
    print("=" * 50)
    
    try:
        # Import the dataloader
        from lmghn3.Dataloader.wikitext2_loader import build_wikitext2
        print("✓ Successfully imported build_wikitext2")
        
        # Test with small parameters for quick testing
        print("\nBuilding dataloader with small parameters...")
        data_config = build_wikitext2(
            tokenizer_name="gpt2",
            seq_len=64,  # Small sequence length for testing
            batch_size=2,  # Small batch size for testing
            num_workers=0,  # No multiprocessing for testing
            cache_dir=None
        )
        print("✓ Successfully built dataloader")
        
        # Check the returned configuration
        print(f"\nDataloader configuration:")
        print(f"  - Vocab size: {data_config['vocab_size']}")
        print(f"  - Tokenizer type: {type(data_config['tokenizer'])}")
        print(f"  - Train loader type: {type(data_config['train_loader'])}")
        print(f"  - Val loader type: {type(data_config['val_loader'])}")
        
        # Test tokenizer
        print(f"\nTesting tokenizer...")
        test_text = "Hello, this is a test sentence."
        tokens = data_config['tokenizer'](test_text)
        print(f"  - Test text: '{test_text}'")
        print(f"  - Tokenized: {tokens['input_ids']}")
        print(f"  - Decoded: '{data_config['tokenizer'].decode(tokens['input_ids'])}'")
        print("✓ Tokenizer working correctly")
        
        # Test train dataloader
        print(f"\nTesting train dataloader...")
        train_loader = data_config['train_loader']
        print(f"  - Number of batches: {len(train_loader)}")
        
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        print(f"  - Sample batch keys: {list(sample_batch.keys())}")
        print(f"  - Input IDs shape: {sample_batch['input_ids'].shape}")
        print(f"  - Attention mask shape: {sample_batch['attention_mask'].shape}")
        print(f"  - Labels shape: {sample_batch['labels'].shape}")
        print(f"  - Input IDs dtype: {sample_batch['input_ids'].dtype}")
        print(f"  - Labels dtype: {sample_batch['labels'].dtype}")
        
        # Check data types and shapes
        expected_seq_len = 64
        expected_batch_size = 2
        
        assert sample_batch['input_ids'].shape == (expected_batch_size, expected_seq_len), \
            f"Expected input_ids shape ({expected_batch_size}, {expected_seq_len}), got {sample_batch['input_ids'].shape}"
        
        assert sample_batch['attention_mask'].shape == (expected_batch_size, expected_seq_len), \
            f"Expected attention_mask shape ({expected_batch_size}, {expected_seq_len}), got {sample_batch['attention_mask'].shape}"
        
        assert sample_batch['labels'].shape == (expected_batch_size, expected_seq_len), \
            f"Expected labels shape ({expected_batch_size}, {expected_seq_len}), got {sample_batch['labels'].shape}"
        
        print("✓ Train dataloader working correctly")
        
        # Test validation dataloader
        print(f"\nTesting validation dataloader...")
        val_loader = data_config['val_loader']
        print(f"  - Number of batches: {len(val_loader)}")
        
        # Get a sample batch from validation
        val_sample_batch = next(iter(val_loader))
        print(f"  - Val batch input IDs shape: {val_sample_batch['input_ids'].shape}")
        print(f"  - Val batch labels shape: {val_sample_batch['labels'].shape}")
        print("✓ Validation dataloader working correctly")
        
        # Test label shifting (next token prediction)
        print(f"\nTesting label shifting for next token prediction...")
        input_ids = sample_batch['input_ids'][0]  # First sequence
        labels = sample_batch['labels'][0]        # Corresponding labels
        
        # Check that labels are shifted by 1 position
        # Labels should be: input_ids[1:], input_ids[2:], ..., input_ids[-1], -100
        for i in range(len(input_ids) - 1):
            if labels[i] != -100:  # Skip padding tokens
                assert labels[i] == input_ids[i + 1], \
                    f"Label at position {i} should be {input_ids[i + 1]}, got {labels[i]}"
        
        # Last position should be -100 (ignore index)
        assert labels[-1] == -100, f"Last label should be -100, got {labels[-1]}"
        print("✓ Label shifting working correctly")
        
        # Test a few more batches to ensure consistency
        print(f"\nTesting multiple batches for consistency...")
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
            
            assert batch['input_ids'].shape == (expected_batch_size, expected_seq_len)
            assert batch['labels'].shape == (expected_batch_size, expected_seq_len)
            assert batch['attention_mask'].shape == (expected_batch_size, expected_seq_len)
        
        print(f"✓ Tested {batch_count} batches - all consistent")
        
        print(f"\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! WikiText-2 dataloader is working correctly.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wikitext_loader()
    sys.exit(0 if success else 1)
