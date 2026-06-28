"""Chuan bi dataset v2 cho Kaggle training."""
import numpy as np
import pandas as pd
import os
import zipfile

RAW_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'gesture_data_v2.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SEQ_LEN = 30
FEAT_DIM = 225

os.makedirs(OUT_DIR, exist_ok=True)
print("Doc CSV...")
df = pd.read_csv(RAW_CSV)
print(f"  {len(df)} dong, {len(df.columns)} cot")

feat_cols = [c for c in df.columns if any(c.startswith(p) for p in ('rh_','lh_','po_'))][:FEAT_DIM]

sequences, emotions, mode_ids = [], [], []
for start in range(0, len(df) - SEQ_LEN + 1, SEQ_LEN):
    chunk = df.iloc[start:start + SEQ_LEN]
    seq   = np.clip(chunk[feat_cols].values.astype(np.float32), -5, 5)
    val   = float(chunk['valence'].mode()[0])
    aro   = float(chunk['arousal'].mode()[0])
    mid   = int(chunk['mode_id'].mode()[0])
    sequences.append(seq)
    emotions.append([val, aro])
    mode_ids.append(mid)

sequences = np.array(sequences, dtype=np.float32)
emotions  = np.array(emotions,  dtype=np.float32)
mode_ids  = np.array(mode_ids,  dtype=np.int32)

print(f"sequences: {sequences.shape}")
print(f"emotions : {emotions.shape}")
print(f"mode_ids : {mode_ids.shape}")

np.save(os.path.join(OUT_DIR, 'sequences.npy'), sequences)
np.save(os.path.join(OUT_DIR, 'emotions.npy'),  emotions)
np.save(os.path.join(OUT_DIR, 'mode_ids.npy'),  mode_ids)

# Zip cho Kaggle
zip_path = os.path.join(OUT_DIR, 'gesturhythm_v2.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(OUT_DIR, 'sequences.npy'), 'sequences.npy')
    zf.write(os.path.join(OUT_DIR, 'emotions.npy'),  'emotions.npy')
    zf.write(os.path.join(OUT_DIR, 'mode_ids.npy'),  'mode_ids.npy')

size = os.path.getsize(zip_path) / 1e6
print(f"\nSaved: gesturhythm_v2.zip ({size:.1f} MB)")
print("Upload file nay len Kaggle Dataset (ten: gesturhythm-v2)")
