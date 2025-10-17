"""
Base classes and common components for language models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union


class BaseConfig:
    """Base configuration class with common parameters."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        max_seq_len: int = 512,
        p_drop: float = 0.1,
        tie_weights: bool = False,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.p_drop = p_drop
        self.tie_weights = tie_weights


class EmbeddingLayer(nn.Module):
    """Common embedding layer with token and optional positional embeddings."""
    
    def __init__(self, cfg: BaseConfig, use_pos_emb: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_pos_emb = use_pos_emb
        
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.p_drop)
        
        if use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        else:
            self.pos_emb = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embeddings."""
        B, T = x.shape
        h = self.tok_emb(x)
        
        if self.pos_emb is not None:
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = h + self.pos_emb(pos)
        
        return self.drop(h)


class LMHead(nn.Module):
    """Language modeling head with optional weight tying."""
    
    def __init__(self, cfg: BaseConfig, input_dim: int, embedding_layer: Optional[nn.Embedding] = None):
        super().__init__()
        self.cfg = cfg
        
        self.lm_head = nn.Linear(input_dim, cfg.vocab_size, bias=False)
        
        # Weight tying
        if cfg.tie_weights and embedding_layer is not None:
            self.lm_head.weight = embedding_layer.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LM head."""
        return self.lm_head(x)


class BaseLanguageModel(nn.Module, ABC):
    """Base class for all language models with common functionality."""
    
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
    
    @abstractmethod
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int = 50, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        
        return idx
    
    def init_weights(self, module: nn.Module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class RNNBaseLanguageModel(BaseLanguageModel):
    """Base class for RNN-based language models (RNN, LSTM, GRU)."""
    
    def __init__(self, cfg: BaseConfig, hidden_size: Optional[int] = None):
        super().__init__(cfg)
        self.hidden_size = hidden_size or cfg.d_model
        
        # Embedding layer
        self.embedding = EmbeddingLayer(cfg, use_pos_emb=False)
        
        # Optional projection layer
        self.proj_in = None
        if cfg.d_model != self.hidden_size:
            self.proj_in = nn.Linear(cfg.d_model, self.hidden_size)
        
        # Dropout
        self.drop_out = nn.Dropout(cfg.p_drop)
        
        # LM head
        self.lm_head = LMHead(cfg, self.hidden_size, self.embedding.tok_emb)
    
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initialize hidden state - to be overridden by subclasses."""
        device = device or next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding layer."""
        h = self.embedding(x)
        if self.proj_in is not None:
            h = self.proj_in(h)
        return h
    
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output head."""
        x = self.drop_out(x)
        return self.lm_head(x)