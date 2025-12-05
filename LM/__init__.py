"""
Modular language models package.

This package provides a collection of language model architectures with a common interface
and shared components for easy experimentation and comparison.

Available Models:
- Mini GPT: Mini GPT with explicit decoder blocks

Usage:
    from LM import create_model_from_config
    
    # Create a custom model
    model = create_model_from_config("mini_gpt", config_dict)
    
    # Or import specific models directly
    from LM import GPTDecoderLM, MiniGPTConfig
    config = MiniGPTConfig(vocab_size=10000, d_model=256, n_layer=2)
    model = GPTDecoderLM(config)
"""

# Import base classes
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead

# Import individual models
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
    "GPTDecoderLM",
    "MiniGPTConfig",
    
    # Trainer
    "Trainer",

]
