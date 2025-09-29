"""
Language Model Architectures for GHN Training

This file contains all the curated LM architectures available for training the GHN.
Each architecture is a Mini-GPT variant with different configurations.
"""

from typing import List, Dict

def get_all_lm_architectures(vocab_size: int = 50257, max_len: int = 512) -> List[Dict]:
    """
    Returns all available LM architectures for GHN training.
    
    Args:
        vocab_size: Vocabulary size (default: 50257 for GPT-2)
        max_len: Maximum sequence length (default: 512)
        
    Returns:
        List of architecture configurations
    """
    
    def _cfg(name: str, d_model: int, n_layers: int, n_heads: int, mlp_ratio: int, max_len: int = 512, vocab_size: int = 50257) -> Dict:
        # Ensure heads divide model dim
        if d_model % n_heads != 0:
            for h in [16, 12, 8, 6, 4, 3, 2]:
                if h <= d_model and d_model % h == 0:
                    n_heads = h
                    break
            else:
                n_heads = 2
        
        d_ff = mlp_ratio * d_model
        return {
            "name": name,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "max_len": max_len,
            "p_drop": 0.1,
            "total_params": _estimate_params(vocab_size, d_model, n_layers, n_heads, d_ff, max_len)
        }
    
    def _estimate_params(vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_len: int) -> int:
        """Estimate total parameters for the architecture."""
        # Token embedding
        token_emb = vocab_size * d_model
        # Position embedding  
        pos_emb = max_len * d_model
        # Transformer layers
        layer_params = n_layers * (
            # Self-attention: Q, K, V projections + output projection
            4 * d_model * d_model +
            # Layer norms (2 per layer)
            2 * d_model +
            # MLP: up projection + down projection
            d_model * d_ff + d_ff * d_model
        )
        # Final layer norm
        final_ln = d_model
        # LM head (tied with token embedding)
        lm_head = 0  # Tied weights
        
        return token_emb + pos_emb + layer_params + final_ln + lm_head
    
    architectures = [
        # Small models (128d)
        _cfg("mini-gpt-2L-128d-h2-r2", 128, 2, 2, 2, max_len, vocab_size),
        _cfg("mini-gpt-3L-128d-h4-r3", 128, 3, 4, 3, max_len, vocab_size),
        _cfg("mini-gpt-4L-128d-h8-r4", 128, 4, 8, 4, max_len, vocab_size),
        
        # Medium models (256d)
        _cfg("mini-gpt-2L-256d-h4-r2", 256, 2, 4, 2, max_len, vocab_size),
        _cfg("mini-gpt-3L-256d-h8-r3", 256, 3, 8, 3, max_len, vocab_size),
        _cfg("mini-gpt-4L-256d-h8-r4", 256, 4, 8, 4, max_len, vocab_size),
        
        # Large models (384d)
        _cfg("mini-gpt-3L-384d-h6-r2", 384, 3, 6, 2, max_len, vocab_size),
        _cfg("mini-gpt-4L-384d-h8-r3", 384, 4, 8, 3, max_len, vocab_size),
        _cfg("mini-gpt-6L-384d-h6-r4", 384, 6, 6, 4, max_len, vocab_size),
        
        # Extra large models (512d)
        _cfg("mini-gpt-4L-512d-h8-r2", 512, 4, 8, 2, max_len, vocab_size),
        _cfg("mini-gpt-6L-512d-h8-r3", 512, 6, 8, 3, max_len, vocab_size),
    ]
    
    return architectures

def print_architecture_summary():
    """Print a summary of all available architectures."""
    architectures = get_all_lm_architectures()
    
    print("=" * 80)
    print("LANGUAGE MODEL ARCHITECTURES FOR GHN TRAINING")
    print("=" * 80)
    print(f"Total architectures: {len(architectures)}")
    print()
    
    # Group by model size
    small_models = [a for a in architectures if a["d_model"] == 128]
    medium_models = [a for a in architectures if a["d_model"] == 256]
    large_models = [a for a in architectures if a["d_model"] == 384]
    xlarge_models = [a for a in architectures if a["d_model"] == 512]
    
    def print_group(name: str, models: List[Dict]):
        if not models:
            return
        print(f"{name.upper()} MODELS ({len(models)} variants):")
        print("-" * 60)
        for arch in models:
            print(f"  {arch['name']:<25} | "
                  f"{arch['n_layers']}L | "
                  f"{arch['d_model']}d | "
                  f"{arch['n_heads']}h | "
                  f"MLP×{arch['d_ff']//arch['d_model']} | "
                  f"{arch['total_params']:,} params")
        print()
    
    print_group("Small (128d)", small_models)
    print_group("Medium (256d)", medium_models)
    print_group("Large (384d)", large_models)
    print_group("Extra Large (512d)", xlarge_models)
    
    # Parameter count summary
    total_params = sum(a["total_params"] for a in architectures)
    avg_params = total_params // len(architectures)
    min_params = min(a["total_params"] for a in architectures)
    max_params = max(a["total_params"] for a in architectures)
    
    print("PARAMETER STATISTICS:")
    print("-" * 30)
    print(f"  Total parameters across all models: {total_params:,}")
    print(f"  Average parameters per model: {avg_params:,}")
    print(f"  Smallest model: {min_params:,} parameters")
    print(f"  Largest model: {max_params:,} parameters")
    print()
    
    # Architecture diversity
    print("ARCHITECTURE DIVERSITY:")
    print("-" * 30)
    layers = sorted(set(a["n_layers"] for a in architectures))
    dims = sorted(set(a["d_model"] for a in architectures))
    heads = sorted(set(a["n_heads"] for a in architectures))
    mlp_ratios = sorted(set(a["d_ff"] // a["d_model"] for a in architectures))
    
    print(f"  Layer counts: {layers}")
    print(f"  Hidden dimensions: {dims}")
    print(f"  Attention heads: {heads}")
    print(f"  MLP ratios: {mlp_ratios}")
    print()
    
    print("=" * 80)

def get_architecture_by_name(name: str) -> Dict:
    """Get a specific architecture by name."""
    architectures = get_all_lm_architectures()
    for arch in architectures:
        if arch["name"] == name:
            return arch
    raise ValueError(f"Architecture '{name}' not found")

def get_architectures_by_size(d_model: int) -> List[Dict]:
    """Get all architectures with a specific hidden dimension."""
    architectures = get_all_lm_architectures()
    return [arch for arch in architectures if arch["d_model"] == d_model]

def get_architectures_by_layers(n_layers: int) -> List[Dict]:
    """Get all architectures with a specific number of layers."""
    architectures = get_all_lm_architectures()
    return [arch for arch in architectures if arch["n_layers"] == n_layers]

if __name__ == "__main__":
    print_architecture_summary()
