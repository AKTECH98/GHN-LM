import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead


# -------------------------------
# Config dataclass
# -------------------------------
class GPTConfig(BaseConfig):
    def __init__(self, vocab_size=32000, d_model=256, n_layer=8, n_head=4, d_ff=1024,
                 max_seq_len=512, p_drop=0.1, attn_drop=0.1, tie_weights=False):
        super().__init__(vocab_size, d_model, max_seq_len, p_drop, tie_weights)
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff
        self.attn_drop = attn_drop


# -------------------------------
# Explicit GPT-style decoder block
# -------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.resid_drop = nn.Dropout(cfg.p_drop)

        mask = torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.p_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTDecoderLM(BaseLanguageModel):
    def __init__(self, cfg: GPTConfig):
        super().__init__(cfg)
        
        # Embeddings
        self.embedding = EmbeddingLayer(cfg, use_pos_emb=True)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        
        # LM head
        self.lm_head = LMHead(cfg, cfg.d_model, self.embedding.tok_emb)

        self.apply(self.init_weights)

    def forward(self, x, targets=None):
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.cfg.max_seq_len}")
        
        # Forward through embeddings
        h = self.embedding(x)
        
        # Forward through transformer blocks
        for blk in self.blocks:
            h = blk(h)
        
        h = self.ln_f(h)
        logits = self.lm_head(h)

        # Compute loss if targets provided
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            return logits, loss
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        return super().generate(idx, max_new_tokens, temperature, top_k)
