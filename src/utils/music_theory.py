import numpy as np
import torch

SCALES = {
    'C_major':    [60,62,64,65,67,69,71,72],
    'C_minor':    [60,62,63,65,67,68,70,72],
    'C_dominant': [60,62,64,65,67,69,70,72],
    'C_pentatonic_major': [60,62,64,67,69,72],
    'C_pentatonic_minor': [60,63,65,67,70,72],
}

def emotion_to_scale(valence, arousal):
    """Chuyen vector cam xuc sang ten scale."""
    if valence > 0.3 and arousal > 0.3:  return 'C_pentatonic_major'
    if valence > 0.3:                     return 'C_major'
    if valence < -0.3 and arousal > 0.3:  return 'C_dominant'
    if valence < -0.3:                    return 'C_minor'
    return 'C_pentatonic_major'

def apply_scale_mask(logits, scale_name, root=60):
    """Mask cac note khong hop le trong scale. logits: (vocab_size,) or (B, vocab_size)."""
    scale_pcs = set(p % 12 for p in SCALES.get(scale_name, SCALES['C_major']))
    mask = torch.ones(128, dtype=torch.bool, device=logits.device)
    for i in range(128):
        if (i - root) % 12 not in scale_pcs:
            mask[i] = False
    # Giu REST (128) va PAD (129) nguyen
    if logits.dim() == 1:
        logits[:128][~mask] = float('-inf')
    else:
        logits[..., :128][..., ~mask] = float('-inf')
    return logits

def snap_to_scale(pitch, scale_name, root=60):
    """Snap pitch ve note gan nhat trong scale."""
    scale_notes = [p + 12*o for o in range(-1,3) for p in SCALES.get(scale_name, SCALES['C_major'])]
    scale_notes = sorted(set(n for n in scale_notes if 0 <= n <= 127))
    return min(scale_notes, key=lambda p: abs(p - pitch))

def humanize(notes, velocity_std=5, timing_std=0.02):
    """Them variation nho vao velocity va timing de co cam giac con nguoi danh."""
    result = []
    for pitch, velocity, duration in notes:
        v = int(np.clip(velocity + np.random.normal(0, velocity_std), 20, 127))
        d = float(np.clip(duration + np.random.normal(0, timing_std), 0.05, 2.0))
        result.append((pitch, v, d))
    return result
