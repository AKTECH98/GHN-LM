# gru_lm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConfig, RNNBaseLanguageModel


class GRUConfig(BaseConfig):
    def __init__(
        self,
        vocab_size=32000,
        d_model=256,          # embedding size (and default hidden size)
        hidden_size=None,     # if None -> d_model
        n_layer=2,
        max_seq_len=512,
        p_drop=0.1,
        tie_weights=False,
    ):
        super().__init__(vocab_size, d_model, max_seq_len, p_drop, tie_weights)
        self.hidden_size = hidden_size or d_model
        self.n_layer = n_layer


class GRULanguageModel(RNNBaseLanguageModel):
    """
    GRU language model:
      Embedding -> (optional proj) -> GRU -> Dropout -> LM head
    """
    def __init__(self, cfg: GRUConfig):
        super().__init__(cfg, cfg.hidden_size)
        
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=cfg.n_layer,
            dropout=cfg.p_drop if cfg.n_layer > 1 else 0.0,
            batch_first=True,
        )
        
        # Initialize weights
        self.apply(self.init_weights)

    def init_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        # (num_layers, batch, hidden)
        return torch.zeros(self.cfg.n_layer, batch_size, self.hidden_size, device=device)

    def forward(self, x, targets=None, hidden=None):
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.cfg.max_seq_len}")

        # Forward through embedding
        h = self.forward_embedding(x)                        # (B, T, H)

        # Forward through GRU
        out, new_hidden = self.gru(h, hidden)                # (B, T, H)
        
        # Forward through output head
        logits = self.forward_head(out)                      # (B, T, V)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
        
        return logits, loss, new_hidden

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        B = idx.size(0)
        hidden = self.init_hidden(B, idx.device)
        for _ in range(max_new_tokens):
            x_last = idx[:, -1:]
            emb = self.forward_embedding(x_last)             # (B, 1, H)
            out, hidden = self.gru(emb, hidden)              # (B, 1, H)
            logits = self.forward_head(out[:, -1, :]) / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
