import torch, torch.nn as nn
import warnings

# Suppress transformer warnings
warnings.filterwarnings('ignore', message='enable_nested_tensor is True, but self.use_nested_tensor is False')

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, n_layers=8, n_heads=4, d_ff=1024, max_len=1024, p_drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=p_drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Remove weight tying to avoid issues with GHN parameter prediction
        # self.lm_head.weight = self.tok.weight  # weight tying

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = self.encoder(h)
        h = self.ln_f(h)
        return self.lm_head(h)
