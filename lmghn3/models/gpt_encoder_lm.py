# gpt_encoder_variant.py
# GPT-style decoder-only LM built from nn.TransformerEncoderLayer + causal mask.
# - Pre-LayerNorm (norm_first=True)
# - GELU MLP
# - Learned absolute positional embeddings
# - Optional weight tying
# - generate() helper for sampling

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConfig, BaseLanguageModel, EmbeddingLayer, LMHead


class GPTConfig(BaseConfig):
    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        n_layer=8,
        n_head=4,
        d_ff=1024,
        max_seq_len=512,
        p_drop=0.1,
        tie_weights=False,
    ):
        super().__init__(vocab_size, d_model, max_seq_len, p_drop, tie_weights)
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff


class GPTEncoderLayerLM(BaseLanguageModel):
    """
    Decoder-only LM behavior via TransformerEncoder + causal mask.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__(cfg)

        # Embeddings
        self.embedding = EmbeddingLayer(cfg, use_pos_emb=True)

        # Encoder stack (pre-LN + GELU)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.p_drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN like GPT-2
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layer)

        # Output head
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = LMHead(cfg, cfg.d_model, self.embedding.tok_emb)

        # Precompute a max-length causal mask (T x T, -inf above the diagonal)
        mask_bool = torch.triu(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1
        )
        attn_mask = mask_bool.float().masked_fill(mask_bool, float("-inf"))
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        # Init (simple GPT-2-like)
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None):
        """
        x: (B, T) token ids
        targets: optional (B, T) with -100 as ignore_index
        pad_mask: optional (B, T) bool, True where tokens are padding (excluded from attention)
        """
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.cfg.max_seq_len}")

        # Forward through embeddings
        h = self.embedding(x)

        # causal attention (no lookahead)
        h = self.encoder(
            h,
            mask=self.attn_mask[:T, :T],
            src_key_padding_mask=pad_mask,  # (B, T) True = ignore
        )

        h = self.ln_f(h)
        logits = self.lm_head(h)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens=50, temperature=1.0, top_k: int | None = None):
        """
        Simple autoregressive sampler.
        idx: (B, T0) start tokens
        """
        return super().generate(idx, max_new_tokens, temperature, top_k)


# Quick sanity usage (remove or guard under __main__ in your repo)
if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=32000, d_model=256, n_layer=4, n_head=4, d_ff=1024, max_seq_len=128)
    model = GPTEncoderLayerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -100
    logits, loss = model(x, targets=y)
    print("params:", sum(p.numel() for p in model.parameters()), "loss:", float(loss))
