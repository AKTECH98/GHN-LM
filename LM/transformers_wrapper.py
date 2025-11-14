"""
Hugging Face Transformers wrapper for GHN training.

This module provides wrappers around Hugging Face Transformers models
to make them compatible with the GHN training system.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from transformers import AutoConfig, AutoModelForCausalLM
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
    
    def __init__(
        self,
        model_name: str = None,
        cfg: TransformersConfig = None,
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
        """
        Initialize TransformersLM.
        
        Can be called in two ways:
        1. With a config object: TransformersLM(cfg=TransformersConfig(...))
        2. With direct parameters: TransformersLM(model_name="gpt2", vocab_size=50257, ...)
        """
        # If cfg is provided, use it; otherwise create from kwargs
        if cfg is not None:
            if model_name is not None or any([vocab_size, d_model, n_layer, n_head, d_ff, max_seq_len, p_drop]):
                raise ValueError("Cannot provide both cfg and individual parameters. Use either cfg or individual parameters.")
            config = cfg
        else:
            if model_name is None:
                raise ValueError("model_name is required. Provide either cfg or model_name parameter.")
            # Create config from kwargs
            config = TransformersConfig(
                model_name=model_name,
                vocab_size=vocab_size,
                d_model=d_model,
                n_layer=n_layer,
                n_head=n_head,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                p_drop=p_drop,
                tie_weights=tie_weights,
                **kwargs
            )
        
        super().__init__(config)
        
        # Validate that d_model is divisible by n_head before creating the model
        if config.d_model is not None and config.n_head is not None:
            if config.d_model % config.n_head != 0:
                raise ValueError(
                    f"`d_model` must be divisible by `n_head` "
                    f"(got `d_model`: {config.d_model} and `n_head`: {config.n_head}). "
                    f"Please ensure d_model % n_head == 0."
                )
        
        # Load the model configuration
        self.hf_config = AutoConfig.from_pretrained(config.model_name, **config.kwargs)
        
        # Override config parameters if specified
        if config.vocab_size is not None:
            self.hf_config.vocab_size = config.vocab_size
        if config.d_model is not None:
            self.hf_config.n_embd = config.d_model
            self.hf_config.d_model = config.d_model
        if config.n_layer is not None:
            self.hf_config.n_layer = config.n_layer
        if config.n_head is not None:
            self.hf_config.n_head = config.n_head
        if config.d_ff is not None:
            self.hf_config.n_inner = config.d_ff
        if config.max_seq_len is not None:
            self.hf_config.n_positions = config.max_seq_len
            self.hf_config.max_position_embeddings = config.max_seq_len
        if config.p_drop is not None:
            self.hf_config.attn_pdrop = config.p_drop
            self.hf_config.embd_pdrop = config.p_drop
            self.hf_config.resid_pdrop = config.p_drop
        
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
    
    def named_parameters(self, *args, **kwargs):
        """Return named parameters for GHN compatibility."""
        return self.model.named_parameters(*args, **kwargs)
    
    def modules(self):
        """Return model modules for GHN compatibility."""
        return self.model.modules()
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Return named modules for GHN compatibility."""
        return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
    
    def named_buffers(self, *args, **kwargs):
        """Return named buffers for GHN compatibility."""
        return self.model.named_buffers(*args, **kwargs)
    
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
    return TransformersLM(model_name=model_name, **kwargs)
