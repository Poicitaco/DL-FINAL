import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GestureEmotionDataset(Dataset):
    """
    Doc gesture_data_v2.csv.
    Return: (sequence (30, 225), emotion (2,))
    """
    SEQ_LEN  = 30
    FEAT_DIM = 225  # 63 rh + 63 lh + 99 pose

    def __init__(self, csv_path, augment=False):
        df      = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if any(
            c.startswith(p) for p in ('rh_', 'lh_', 'po_'))][:self.FEAT_DIM]
        self.sequences, self.emotions = [], []
        for start in range(0, len(df) - self.SEQ_LEN + 1, self.SEQ_LEN):
            chunk = df.iloc[start:start + self.SEQ_LEN]
            seq   = chunk[feat_cols].values.astype(np.float32)
            val   = float(chunk['valence'].mode()[0])
            aro   = float(chunk['arousal'].mode()[0])
            self.sequences.append(seq)
            self.emotions.append([val, aro])
        self.augment = augment

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        emo = torch.tensor(self.emotions[idx],  dtype=torch.float32)
        if self.augment:
            seq = seq + torch.randn_like(seq) * 0.01  # gaussian noise
        return seq, emo


class MusicPriorDataset(Dataset):
    """
    Doc MIDI token sequences.
    Return: (input_tokens (T,), target_tokens (T,)) cho language model training.
    """
    def __init__(self, token_sequences, seq_len=128):
        self.data    = token_sequences  # list of token lists
        self.seq_len = seq_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Pad hoac crop ve seq_len
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [129] * (self.seq_len + 1 - len(tokens))  # 129=PAD
        tokens = tokens[:self.seq_len + 1]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        return x, y
