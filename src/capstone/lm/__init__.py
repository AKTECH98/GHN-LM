"""
Modular language models package.

Usage:
    from capstone.lm.create_model import create_model
    from capstone.data.config_loader import load_config_file

    model_config, _, _ = load_config_file("configs/benchmarks/benchmark_1_tiny.yaml")
    model = create_model(model_config, vocab_size=50257)
"""

from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead
from .gpt_encoder_lm import GPTEncoderLayerLM, GPTConfig as GPTEncoderConfig
from .mini_gpt import GPTDecoderLM, GPTConfig as MiniGPTConfig
from .trainer import Trainer

__all__ = [
    "BaseConfig",
    "BaseLanguageModel",
    "EmbeddingLayer",
    "LMHead",
    "GPTEncoderLayerLM",
    "GPTEncoderConfig",
    "GPTDecoderLM",
    "MiniGPTConfig",
    "Trainer",
]
