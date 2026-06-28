import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicPrior(nn.Module):
    """
    GPT-style autoregressive model hoc ngu phap am nhac.
    Hoc P(note_t | note_0..t-1) tu MIDI dataset.
    Input:  (B, T) - token sequence (0-127=note, 128=REST, 129=PAD)
    Output: (B, T, 130) - logits cho next token
    """
    VOCAB  = 130  # 128 notes + REST + PAD
    REST   = 128
    PAD    = 129

    def __init__(self, vocab_size=130, d_model=128, nhead=4, num_layers=4, max_len=512):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=self.PAD)
        self.pos_enc = nn.Embedding(max_len, d_model)
        dec_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.head        = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embed(tokens) + self.pos_enc(pos)
        x    = self.transformer(x, mask=mask)
        return self.head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, primer_tokens, max_len=64, temperature=1.0):
        """Autoregressive generation."""
        self.eval()
        tokens = primer_tokens.clone()
        for _ in range(max_len - tokens.shape[1]):
            logits  = self.forward(tokens)[:, -1, :] / temperature
            probs   = F.softmax(logits, dim=-1)
            next_t  = torch.multinomial(probs, num_samples=1)
            tokens  = torch.cat([tokens, next_t], dim=1)
        return tokens
