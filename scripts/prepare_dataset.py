"""
Sinh melody thật tu nhac ly de lam target cho model.
Thay vi rule don gian, dung cac progression va pattern
co trong nhac thuc te (pop, ballad, jazz).
Ket hop voi gesture data theo mood.
"""
import numpy as np
import pandas as pd
import os
import random

random.seed(42)
np.random.seed(42)

RAW_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'gesture_data.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SEQ_LEN = 30
NOTE_OUT = 16

PITCH_MIN, PITCH_MAX = 60, 72
VEL_MIN,   VEL_MAX   = 40, 127
TEMPO_MIN, TEMPO_MAX = 80, 160

# ── Melody patterns thực tế từ nhạc pop/ballad/jazz ──────────────────────
# Format: (interval_pattern, velocity_contour, duration_pattern)
# interval = so ban cung so voi note truoc (0=giu, 1=len 1, -1=xuong 1...)

REAL_MELODIES = {
    "calm_major": [
        # Twinkle twinkle style
        ([0,0,7,7,9,9,7,0,5,5,4,4,2,2,0], [70,70,75,75,75,75,70,65,70,70,65,65,65,65,60], [1,1,1,1,1,1,2,1,1,1,1,1,1,1,2]),
        # Ode to Joy style
        ([0,0,2,4,4,2,0,-1,-3,-3,-1,0,0,-1,-1], [70,70,72,75,75,72,70,68,65,65,68,70,70,68,65], [1,1,1,1,1,1,1,1,1,1,1,1,2,1,2]),
        # Simple ascending
        ([0,2,4,5,7,5,4,2,0,-1,0,2,4,2,0], [65,68,70,72,75,72,70,68,65,63,65,68,70,68,65], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]),
    ],
    "energetic_major": [
        # Upbeat jumping
        ([0,4,7,4,0,4,7,12,7,4,0,-1,0,2,4], [80,85,90,85,80,85,90,95,90,85,80,75,80,82,85], [0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5,0.5,0.5,1]),
        # Fast run
        ([0,2,4,5,7,9,11,12,11,9,7,5,4,2,0], [75,78,80,82,85,87,90,95,90,87,85,82,80,78,75], [0.5]*15),
        # Rock riff style
        ([0,0,5,0,0,7,0,5,3,0,0,5,0,0,3], [85,82,90,82,85,92,82,88,85,82,85,88,82,85,80], [0.5,0.5,1,0.5,0.5,1,0.5,1,0.5,0.5,0.5,1,0.5,0.5,2]),
    ],
    "sad_minor": [
        # Descending minor
        ([0,-2,-3,-5,-7,-5,-3,-2,0,-2,-3,-5,-3,-2,0], [70,68,65,63,60,63,65,68,65,63,60,58,60,63,65], [1,1,1,1,2,1,1,1,1,1,1,1,1,1,2]),
        # Melancholic
        ([0,-1,-3,-5,-3,-1,0,-3,-5,-7,-5,-3,-1,0,-2], [65,63,60,58,60,63,65,60,58,55,58,60,63,65,62], [1,1,2,1,1,1,1,1,2,1,1,1,1,2,1]),
        # Bach-like descending
        ([0,-2,-4,-5,-7,-5,-4,-2,0,-1,-3,-5,-4,-2,0], [68,65,63,60,58,60,63,65,68,66,63,60,63,65,68], [1,1,1,1,2,1,1,1,1,1,1,1,1,1,2]),
    ],
    "tense_dominant": [
        # Blues feel
        ([0,3,5,6,7,6,5,3,0,3,5,7,5,3,0], [80,82,85,87,90,87,85,82,78,82,85,90,85,82,78], [0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,1]),
        # Jazz influenced
        ([0,2,4,6,5,3,2,0,-1,1,3,5,3,1,0], [75,78,80,85,82,80,78,75,73,75,78,83,80,78,75], [1]*15),
    ],
}

def get_mood(pitch, velocity, tempo, chord):
    """Xac dinh mood tu labels."""
    if chord in [1] and tempo > 110:  return "energetic_major"
    if chord in [1, 0] and tempo <= 110: return "calm_major"
    if chord in [2]:                  return "sad_minor"
    if chord in [3]:                  return "tense_dominant"
    return "calm_major"

def melody_to_notes(root_pitch, intervals, velocities, durations):
    """Chuyen pattern thanh sequence notes normalized."""
    notes = []
    current = root_pitch
    for i in range(min(NOTE_OUT, len(intervals))):
        if i > 0:
            current = np.clip(current + intervals[i], PITCH_MIN, PITCH_MAX)
        vel = np.clip(velocities[i], VEL_MIN, VEL_MAX)
        dur = durations[i]
        notes.append([
            (current - PITCH_MIN) / (PITCH_MAX - PITCH_MIN),
            (vel      - VEL_MIN)  / (VEL_MAX   - VEL_MIN),
            min(dur / 2.0, 1.0)
        ])
    # Pad neu thieu
    while len(notes) < NOTE_OUT:
        notes.append(notes[-1] if notes else [0.5, 0.5, 0.5])
    return np.array(notes[:NOTE_OUT], dtype=np.float32)

def generate_real_note_sequence(pitch, velocity, tempo, chord):
    mood     = get_mood(pitch, velocity, tempo, chord)
    patterns = REAL_MELODIES[mood]
    intervals, velocities, durations = random.choice(patterns)
    # Transpose root den pitch label
    root = int(np.clip(pitch, PITCH_MIN, PITCH_MAX))
    # Them variation nho de data da dang
    vel_var = np.random.randint(-8, 8, len(velocities))
    velocities = [v + dv for v, dv in zip(velocities, vel_var)]
    return melody_to_notes(root, intervals, velocities, durations)

# ── Main ──────────────────────────────────────────────────────────────────
print("Doc CSV...")
df = pd.read_csv(RAW_CSV)
print(f"  {len(df)} dong, {len(df.columns)} cot")

lm_cols = [c for c in df.columns if c.startswith('r_lm') or c.startswith('l_lm')]
label_cols = ['pitch','velocity','tempo','chord_type','right_present','left_present']

print("Tach sequences va sinh real melodies...")
sequences, note_sequences, labels = [], [], []

for start in range(0, len(df) - SEQ_LEN + 1, SEQ_LEN):
    chunk = df.iloc[start:start + SEQ_LEN]
    both_off = ((chunk['right_present'] == 0) & (chunk['left_present'] == 0)).sum()
    if both_off > 20: continue

    seq = np.clip(chunk[lm_cols].values.astype(np.float32), 0, 1)
    lbl = {col: chunk[col].mode()[0] for col in label_cols}

    # Sinh real melody thay vi rule
    note_seq = generate_real_note_sequence(
        int(lbl['pitch']), int(lbl['velocity']),
        int(lbl['tempo']),  int(lbl['chord_type'])
    )

    sequences.append(seq)
    note_sequences.append(note_seq)
    labels.append([lbl['pitch'], lbl['velocity'], lbl['tempo'],
                   lbl['chord_type'], lbl['right_present'], lbl['left_present']])

sequences     = np.array(sequences,     dtype=np.float32)
note_sequences = np.array(note_sequences, dtype=np.float32)
labels         = np.array(labels,        dtype=np.float32)

print(f"  Sequences : {sequences.shape}")
print(f"  Melodies  : {note_sequences.shape}")
print(f"  Labels    : {labels.shape}")

# Luu
os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, 'sequences.npy'),      sequences)
np.save(os.path.join(OUT_DIR, 'note_sequences.npy'), note_sequences)
np.save(os.path.join(OUT_DIR, 'labels.npy'),         labels)

# Zip cho Kaggle
import zipfile
zip_path = os.path.join(OUT_DIR, 'gesturhythm_dataset.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(OUT_DIR, 'sequences.npy'),      'sequences.npy')
    zf.write(os.path.join(OUT_DIR, 'note_sequences.npy'), 'note_sequences.npy')
    zf.write(os.path.join(OUT_DIR, 'labels.npy'),         'labels.npy')

size = os.path.getsize(zip_path) / 1e6
print(f"\nDa luu: {zip_path} ({size:.1f} MB)")
print("Upload gesturhythm_dataset.zip len Kaggle.")

# Thong ke mood
from collections import Counter
moods = [get_mood(int(labels[i,0]), int(labels[i,1]), int(labels[i,2]), int(labels[i,3]))
         for i in range(len(labels))]
print("\nPhan bo mood:")
for m, c in Counter(moods).most_common():
    print(f"  {m}: {c} samples")
