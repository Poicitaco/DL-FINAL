import torch
import torch.nn as nn


class ConditionedDecoder(nn.Module):
    """
    Cross-Attention decoder: emotion vector guide music generation.
    emotion_vec (B,2) lam Key/Value, music context lam Query.
    Output: logits (B, T, 130) cho next token prediction.
    """
    def __init__(self, emotion_dim=2, d_model=128, nhead=4, num_layers=2, vocab_size=130):
        super().__init__()
        self.emotion_proj = nn.Linear(emotion_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.head        = nn.Linear(d_model, vocab_size)

    def forward(self, emotion_vec, music_prior_output):
        """
        emotion_vec:        (B, 2)
        music_prior_output: (B, T, d_model) - output tu MusicPrior transformer
        """
        # Emotion lam memory (Key/Value): (B, 1, d_model)
        memory = self.emotion_proj(emotion_vec).unsqueeze(1)
        # Music context lam Query
        out    = self.transformer(music_prior_output, memory)  # (B, T, d_model)
        return self.head(out)  # (B, T, vocab_size)
