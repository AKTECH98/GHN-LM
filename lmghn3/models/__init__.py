"""
Modular language models package.

This package provides a collection of language model architectures with a common interface
and shared components for easy experimentation and comparison.

Available Models:
- RNN: Plain RNN (tanh) language model
- LSTM: LSTM language model  
- GRU: GRU language model
- GPT Encoder: GPT-style model using TransformerEncoder with causal mask
- Mini GPT: Mini GPT with explicit decoder blocks

Usage:
    from models import create_model
    
    # Create a model using the factory
    model = create_model("lstm", vocab_size=10000, d_model=256, n_layer=2)
    
    # Or import specific models directly
    from models import LSTMLanguageModel, LSTMConfig
    config = LSTMConfig(vocab_size=10000, d_model=256, n_layer=2)
    model = LSTMLanguageModel(config)
"""

# Import base classes
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead, RNNBaseLanguageModel

# Import individual models
from .gpt_encoder_lm import GPTEncoderLayerLM, GPTConfig as GPTEncoderConfig
from .mini_gpt import GPTDecoderLM, GPTConfig as MiniGPTConfig

# Define what gets imported with "from models import *"
__all__ = [
    # Base classes
    "BaseConfig",
    "BaseLanguageModel", 
    "EmbeddingLayer",
    "LMHead",
    "RNNBaseLanguageModel",
    
    # Model classes
    "GPTEncoderLayerLM",
    "GPTEncoderConfig",
    "GPTDecoderLM",
    "MiniGPTConfig",

]
