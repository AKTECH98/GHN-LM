"""
Modular language models package.

This package provides a collection of language model architectures with a common interface
and shared components for easy experimentation and comparison.

Available Models:
- GPT Encoder: GPT-style model using TransformerEncoder with causal mask
- Mini GPT: Mini GPT with explicit decoder blocks
- Transformers Wrapper: HuggingFace Transformers models (GPT-2, GPT-Neo, Mistral, etc.)

Usage:
    from LM import create_model_from_config
    
    # Create a custom model
    model = create_model_from_config("gpt_encoder", config_dict)
    
    # Create a HuggingFace model
    model = create_model_from_config("transformers", config_dict)
    
    # Or import specific models directly
    from LM import GPTEncoderLayerLM, GPTEncoderConfig, TransformersLM
    config = GPTEncoderConfig(vocab_size=10000, d_model=256, n_layer=2)
    model = GPTEncoderLayerLM(config)
"""

# Import base classes
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead

# Import individual models
from .gpt_encoder_lm import GPTEncoderLayerLM, GPTConfig as GPTEncoderConfig
from .mini_gpt import GPTDecoderLM, GPTConfig as MiniGPTConfig
from .transformers_wrapper import TransformersLM, TransformersConfig, create_transformers_model, OSS_MODEL_CONFIGS

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
    "TransformersLM",
    "TransformersConfig",
    "create_transformers_model",
    "OSS_MODEL_CONFIGS",
    
    # Trainer
    "Trainer",

]
