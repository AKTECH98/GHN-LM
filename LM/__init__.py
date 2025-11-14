"""
Modular language models package.

This package provides a collection of language model architectures with a common interface
and shared components for easy experimentation and comparison.

Available Models:
- Transformers Wrapper: HuggingFace Transformers models (GPT-2, GPT-Neo, Mistral, etc.)

Usage:
    from LM import TransformersLM
    
    # Create a HuggingFace model (simplified - no config needed)
    model = TransformersLM(model_name="gpt2", vocab_size=50257)
"""

# Import base classes
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead

# Import individual models
from .transformers_wrapper import TransformersLM, TransformersConfig, create_transformers_model

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
    "TransformersLM",
    "TransformersConfig",
    "create_transformers_model",
    
    # Trainer
    "Trainer",

]
