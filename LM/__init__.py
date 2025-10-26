"""
Modular language models package.

This package provides a collection of language model architectures with a common interface
and shared components for easy experimentation and comparison.

Available Models:
- GPT Encoder: GPT-style model using TransformerEncoder with causal mask
- Mini GPT: Mini GPT with explicit decoder blocks

Usage:
    from models import create_model
    
    # Create a model using the factory
    model = create_model("gpt_encoder", vocab_size=10000, d_model=256, n_layer=2)
    
    # Or import specific models directly
    from models import GPTEncoderLayerLM, GPTEncoderConfig
    config = GPTEncoderConfig(vocab_size=10000, d_model=256, n_layer=2)
    model = GPTEncoderLayerLM(config)
"""

# Import base classes
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead

# Import individual models
from .gpt_encoder_lm import GPTEncoderLayerLM, GPTConfig as GPTEncoderConfig
from .mini_gpt import GPTDecoderLM, GPTConfig as MiniGPTConfig

# Import trainer
from .trainer import Trainer

# Define what gets imported with "from models import *"
__all__ = [
    # Base classes
    "BaseConfig",
    "BaseLanguageModel", 
    "EmbeddingLayer",
    "LMHead",
    
    # Model classes
    "GPTEncoderLayerLM",
    "GPTEncoderConfig",
    "GPTDecoderLM",
    "MiniGPTConfig",
    
    # Trainer
    "Trainer",

]
