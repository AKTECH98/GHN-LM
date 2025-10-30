"""
Hugging Face Transformers wrapper for GHN training.

This module provides wrappers around Hugging Face Transformers models
to make them compatible with the GHN training system.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .base import BaseConfig, BaseLanguageModel


class TransformersConfig(BaseConfig):
    """Configuration for Transformers-based models."""
    
    def __init__(
        self,
        model_name: str,
        vocab_size: int = None,
        d_model: int = None,
        n_layer: int = None,
        n_head: int = None,
        d_ff: int = None,
        max_seq_len: int = None,
        p_drop: float = None,
        tie_weights: bool = False,
        **kwargs
    ):
        super().__init__(vocab_size or 50257, d_model or 768, max_seq_len or 1024, p_drop or 0.1, tie_weights)
        self.model_name = model_name
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff
        self.kwargs = kwargs


class TransformersLM(BaseLanguageModel):
    """Wrapper around Hugging Face Transformers models for GHN training."""
    
    def __init__(self, cfg: TransformersConfig):
        super().__init__(cfg)
        
        # Load the model configuration
        self.hf_config = AutoConfig.from_pretrained(cfg.model_name, **cfg.kwargs)
        
        # Override config parameters if specified
        if cfg.vocab_size is not None:
            self.hf_config.vocab_size = cfg.vocab_size
        if cfg.d_model is not None:
            self.hf_config.n_embd = cfg.d_model
            self.hf_config.d_model = cfg.d_model
        if cfg.n_layer is not None:
            self.hf_config.n_layer = cfg.n_layer
        if cfg.n_head is not None:
            self.hf_config.n_head = cfg.n_head
        if cfg.d_ff is not None:
            self.hf_config.n_inner = cfg.d_ff
        if cfg.max_seq_len is not None:
            self.hf_config.n_positions = cfg.max_seq_len
            self.hf_config.max_position_embeddings = cfg.max_seq_len
        if cfg.p_drop is not None:
            self.hf_config.attn_pdrop = cfg.p_drop
            self.hf_config.embd_pdrop = cfg.p_drop
            self.hf_config.resid_pdrop = cfg.p_drop
        
        # Create the model
        self.model = AutoModelForCausalLM.from_config(self.hf_config)
        
        # Update our config with actual values from HF config
        self._update_config_from_hf()
    
    def _update_config_from_hf(self):
        """Update our config with actual values from HuggingFace config."""
        # Map different config attribute names to our standard names
        if hasattr(self.hf_config, 'n_embd'):
            self.cfg.d_model = self.hf_config.n_embd
        elif hasattr(self.hf_config, 'd_model'):
            self.cfg.d_model = self.hf_config.d_model
        elif hasattr(self.hf_config, 'hidden_size'):
            self.cfg.d_model = self.hf_config.hidden_size
            
        if hasattr(self.hf_config, 'n_layer'):
            self.cfg.n_layer = self.hf_config.n_layer
        elif hasattr(self.hf_config, 'num_hidden_layers'):
            self.cfg.n_layer = self.hf_config.num_hidden_layers
            
        if hasattr(self.hf_config, 'n_head'):
            self.cfg.n_head = self.hf_config.n_head
        elif hasattr(self.hf_config, 'num_attention_heads'):
            self.cfg.n_head = self.hf_config.num_attention_heads
            
        if hasattr(self.hf_config, 'n_inner'):
            self.cfg.d_ff = self.hf_config.n_inner
        elif hasattr(self.hf_config, 'intermediate_size'):
            self.cfg.d_ff = self.hf_config.intermediate_size
            
        if hasattr(self.hf_config, 'n_positions'):
            self.cfg.max_seq_len = self.hf_config.n_positions
        elif hasattr(self.hf_config, 'max_position_embeddings'):
            self.cfg.max_seq_len = self.hf_config.max_position_embeddings
            
        self.cfg.vocab_size = self.hf_config.vocab_size
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model."""
        # Prepare inputs for HuggingFace model
        inputs = {"input_ids": x}
        if targets is not None:
            inputs["labels"] = targets
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Extract logits and loss
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        
        return logits, loss
    
    def parameters(self):
        """Return model parameters for GHN compatibility."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Return named parameters for GHN compatibility."""
        return self.model.named_parameters()
    
    def modules(self):
        """Return model modules for GHN compatibility."""
        return self.model.modules()
    
    def named_modules(self):
        """Return named modules for GHN compatibility."""
        return self.model.named_modules()
    
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    def train(self, mode=True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self


def create_transformers_model(model_name: str, **kwargs) -> TransformersLM:
    """Factory function to create Transformers-based models."""
    config = TransformersConfig(model_name=model_name, **kwargs)
    return TransformersLM(config)


# Predefined model configurations for popular OSS models
OSS_MODEL_CONFIGS = {
    "gpt2-small": {
        "model_name": "gpt2",
        "d_model": 768,
        "n_layer": 12,
        "n_head": 12,
        "d_ff": 3072,
        "max_seq_len": 1024,
    },
    "gpt2-medium": {
        "model_name": "gpt2-medium",
        "d_model": 1024,
        "n_layer": 24,
        "n_head": 16,
        "d_ff": 4096,
        "max_seq_len": 1024,
    },
    "gpt2-large": {
        "model_name": "gpt2-large",
        "d_model": 1280,
        "n_layer": 36,
        "n_head": 20,
        "d_ff": 5120,
        "max_seq_len": 1024,
    },
    "gpt2-xl": {
        "model_name": "gpt2-xl",
        "d_model": 1600,
        "n_layer": 48,
        "n_head": 25,
        "d_ff": 6400,
        "max_seq_len": 1024,
    },
    "gpt-neo-125m": {
        "model_name": "EleutherAI/gpt-neo-125M",
        "d_model": 768,
        "n_layer": 12,
        "n_head": 12,
        "d_ff": 3072,
        "max_seq_len": 2048,
    },
    "gpt-neo-1.3b": {
        "model_name": "EleutherAI/gpt-neo-1.3B",
        "d_model": 2048,
        "n_layer": 24,
        "n_head": 16,
        "d_ff": 8192,
        "max_seq_len": 2048,
    },
    "gpt-neo-2.7b": {
        "model_name": "EleutherAI/gpt-neo-2.7B",
        "d_model": 2560,
        "n_layer": 32,
        "n_head": 20,
        "d_ff": 10240,
        "max_seq_len": 2048,
    },
    "gpt-j-6b": {
        "model_name": "EleutherAI/gpt-j-6B",
        "d_model": 4096,
        "n_layer": 28,
        "n_head": 16,
        "d_ff": 16384,
        "max_seq_len": 2048,
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "d_model": 4096,
        "n_layer": 32,
        "n_head": 32,
        "d_ff": 14336,
        "max_seq_len": 32768,
    },
    "mpt-7b": {
        "model_name": "mosaicml/mpt-7b",
        "d_model": 4096,
        "n_layer": 32,
        "n_head": 32,
        "d_ff": 14336,
        "max_seq_len": 2048,
    },
}
