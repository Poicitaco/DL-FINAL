import torch
import torch.nn as nn


class GestureEmotionEncoder(nn.Module):
    """
    Transformer Encoder hoc vector cam xuc tu chuoi landmarks.
    Input:  (B, T, 225) - T frames, 225 features (21 hand + 33 pose landmarks)
    Output: (B, 2)      - [valence, arousal] in [-1, 1]
    """
    def __init__(self, input_dim=225, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Embedding(512, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head        = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        x    = self.embed(x) + self.pos_enc(pos)
        x    = self.transformer(x, mask=mask)
        return self.head(x[:, -1, :])  # (B, 2)
