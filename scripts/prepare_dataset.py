"""
Chuan bi dataset tu gesture_data.csv cho Kaggle.
- Tach thanh sequences 30 frame
- Normalize landmarks
- Luu ra files npy + labels.csv
"""
import numpy as np
import pandas as pd
import os

RAW_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'gesture_data.csv')
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SEQ_LEN  = 30

os.makedirs(OUT_DIR, exist_ok=True)

print("Doc file CSV...")
df = pd.read_csv(RAW_CSV)
print(f"  {len(df)} dong, {len(df.columns)} cot")

# Cot landmark
lm_cols = [c for c in df.columns if c.startswith('r_lm') or c.startswith('l_lm')]
label_cols = ['pitch', 'velocity', 'tempo', 'chord_type', 'right_present', 'left_present']

print("Tach sequences...")
sequences = []
labels    = []
seq_meta  = []  # luu info de debug

for start in range(0, len(df) - SEQ_LEN + 1, SEQ_LEN):
    chunk = df.iloc[start:start + SEQ_LEN]

    # Bo qua neu qua nhieu frame mat tay (ca 2 tay deu off > 20 frame)
    both_off = ((chunk['right_present'] == 0) & (chunk['left_present'] == 0)).sum()
    if both_off > 20:
        continue

    # Input: (30, 126) - landmarks 2 tay
    seq = chunk[lm_cols].values.astype(np.float32)

    # Normalize: chia cho max de dua ve [0,1]
    # Landmark da o [0,1] tu MediaPipe, chi can xu ly outlier
    seq = np.clip(seq, 0, 1)

    # Label: lay gia tri pho bien nhat trong 30 frame
    lbl = {}
    for col in label_cols:
        lbl[col] = chunk[col].mode()[0]

    sequences.append(seq)
    labels.append([lbl['pitch'], lbl['velocity'], lbl['tempo'],
                   lbl['chord_type'], lbl['right_present'], lbl['left_present']])
    seq_meta.append(start)

sequences = np.array(sequences, dtype=np.float32)  # (N, 30, 126)
labels    = np.array(labels,    dtype=np.float32)   # (N, 6)

print(f"  Sequences shape : {sequences.shape}")
print(f"  Labels shape    : {labels.shape}")

# Luu file
np.save(os.path.join(OUT_DIR, 'sequences.npy'), sequences)
np.save(os.path.join(OUT_DIR, 'labels.npy'),    labels)

# Luu labels.csv de de xem
label_df = pd.DataFrame(labels, columns=label_cols)
label_df.to_csv(os.path.join(OUT_DIR, 'labels.csv'), index=False)

# Thong ke
print("\n=== PHAN BO LABELS ===")
print(f"Pitch  : {np.unique(labels[:,0], return_counts=True)}")
print(f"Chord  : {np.unique(labels[:,3], return_counts=True)}")
print(f"R_pres : {np.unique(labels[:,4], return_counts=True)}")
print(f"L_pres : {np.unique(labels[:,5], return_counts=True)}")

print(f"\nLuu vao {OUT_DIR}")
print("  sequences.npy")
print("  labels.npy")
print("  labels.csv")
print("\nUpload 3 files nay len Kaggle Dataset.")
