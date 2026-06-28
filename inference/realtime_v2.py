"""
GestuRhythm v2 - Real-time inference
Pipeline: Webcam -> MediaPipe Holistic -> GestureEmotionEncoder -> emotion vector
       -> ConditionedDecoder + MusicPrior -> MIDI tokens -> Scale mask -> FluidSynth
"""
import os, sys, time, collections, threading
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fluidsynth

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['PATH'] += r';C:\tools\fluidsynth\bin'

BASE       = os.path.join(os.path.dirname(__file__), '..')
ENC_PATH   = os.path.join(BASE, 'model', 'gesture_emotion_encoder.pt')
PRIOR_PATH = os.path.join(BASE, 'model', 'music_prior.pt')
DEC_PATH   = os.path.join(BASE, 'model', 'conditioned_decoder.pt')
SF_PATH    = os.path.join(BASE, 'soundfonts', 'FluidR3_GM.sf2')
CAM_CONFIG = os.path.join(BASE, 'camera_config.txt')
SEQ_LEN    = 30
FEAT_DIM   = 225

# ── Models ────────────────────────────────────────────────────────────────
class GestureEmotionEncoder(nn.Module):
    def __init__(self, input_dim=225, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Embedding(512, d_model)
        enc_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(),
                                  nn.Linear(64,2), nn.Tanh())
    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        x    = self.embed(x) + self.pos_enc(pos)
        return self.head(self.transformer(x, mask=mask)[:, -1, :])

class MusicPrior(nn.Module):
    def __init__(self, vocab_size=130, d_model=128, nhead=4, num_layers=4, max_len=512):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=129)
        self.pos_enc = nn.Embedding(max_len, d_model)
        enc_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head        = nn.Linear(d_model, vocab_size)
    def forward(self, tokens):
        B, T = tokens.shape
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embed(tokens) + self.pos_enc(pos)
        return self.head(self.transformer(x, mask=mask))
    def get_hidden(self, tokens):
        B, T = tokens.shape
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embed(tokens) + self.pos_enc(pos)
        return self.transformer(x, mask=mask)

class ConditionedDecoder(nn.Module):
    def __init__(self, emotion_dim=2, d_model=128, nhead=4, num_layers=2, vocab_size=130):
        super().__init__()
        self.emotion_proj = nn.Linear(emotion_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, emotion_vec, prior_hidden):
        memory = self.emotion_proj(emotion_vec).unsqueeze(1)
        return self.head(self.transformer(prior_hidden, memory))

def load_models():
    enc_ckpt   = torch.load(ENC_PATH,   map_location='cpu', weights_only=False)
    prior_ckpt = torch.load(PRIOR_PATH, map_location='cpu', weights_only=False)
    encoder = GestureEmotionEncoder(**enc_ckpt['config'])
    encoder.load_state_dict(enc_ckpt['model_state'])
    encoder.eval()
    prior   = MusicPrior(**prior_ckpt['config'])
    prior.load_state_dict(prior_ckpt['model_state'])
    prior.eval()
    # Load ConditionedDecoder neu co
    decoder = None
    if os.path.exists(DEC_PATH):
        dec_ckpt = torch.load(DEC_PATH, map_location='cpu', weights_only=False)
        decoder  = ConditionedDecoder(**dec_ckpt['config'])
        decoder.load_state_dict(dec_ckpt['model_state'])
        decoder.eval()
        print('ConditionedDecoder loaded')
    else:
        print('conditioned_decoder.pt not found - dung bias fallback')
    return encoder, prior, decoder

# ── Music Theory ──────────────────────────────────────────────────────────
SCALES = {
    'major':     [0,2,4,5,7,9,11],
    'minor':     [0,2,3,5,7,8,10],
    'dominant':  [0,2,4,5,7,9,10],
    'pentatonic':[0,2,4,7,9],
}

def emotion_to_config(valence, arousal):
    """Chuyen [valence, arousal] sang cau hinh sinh nhac."""
    if valence > 0.2:
        scale = 'major' if arousal > 0 else 'pentatonic'
        prog, inst = (24, 'Guitar') if arousal > 0.3 else (0, 'Piano')
        tempo = int(np.clip(100 + arousal * 50, 80, 160))
        temp  = max(0.5, 1.2 - arousal * 0.5)
    else:
        scale = 'minor'
        prog, inst = (48, 'Strings') if arousal < 0.2 else (49, 'Strings Fast')
        tempo = int(np.clip(80 + arousal * 30, 60, 120))
        temp  = max(0.8, 1.5 - arousal * 0.3)
    return scale, prog, inst, tempo, temp

def apply_scale_mask(logits, scale_name, root=60):
    """Mask cac note khong hop le -> -inf."""
    scale_pcs = set(SCALES[scale_name])
    for i in range(128):
        if (i - root) % 12 not in scale_pcs:
            logits[i] = float('-inf')
    return logits

def generate_notes(primer_tokens, prior, emotion_vec, scale, temperature, n_notes=16, decoder=None):
    """Sinh n_notes: dung ConditionedDecoder neu co, fallback sang bias neu khong."""
    tokens = torch.tensor([primer_tokens], dtype=torch.long)
    notes  = []
    with torch.no_grad():
        for _ in range(n_notes):
            prior_hidden = prior.get_hidden(tokens)  # (1,T,128)
            if decoder is not None:
                # Dung Cross-Attention that su
                emo  = emotion_vec.unsqueeze(0) if emotion_vec.dim() == 1 else emotion_vec
                logits = decoder(emo, prior_hidden)[0, -1, :].clone()
            else:
                # Fallback: bias logits thu cong
                logits = prior(tokens)[0, -1, :].clone()
                valence, arousal = float(emotion_vec[0]), float(emotion_vec[1])
                pitch_bias = np.zeros(128)
                center = int(np.clip(64 + valence * 12 + arousal * 6, 48, 84))
                for i in range(128):
                    pitch_bias[i] = -abs(i - center) * 0.05
                logits[:128] += torch.tensor(pitch_bias, dtype=torch.float32)
            # Apply scale mask
            logits_np = logits.numpy().copy()
            logits_np = apply_scale_mask(logits_np, scale)
            logits    = torch.tensor(logits_np)
            probs  = F.softmax(logits / temperature, dim=0)
            next_t = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_t.unsqueeze(0)], dim=1)
            if next_t.item() < 128:
                notes.append(next_t.item())
    return notes if notes else [60, 64, 67]
    """Sinh n_notes tu Music Prior, conditioned boi emotion."""
    tokens = torch.tensor([primer_tokens], dtype=torch.long)
    notes  = []
    with torch.no_grad():
        for _ in range(n_notes):
            logits = prior(tokens)[0, -1, :].clone()  # (130,)
            # Emotion conditioning: bias pitch range theo valence/arousal
            valence, arousal = float(emotion_vec[0]), float(emotion_vec[1])
            # Valence > 0: uu tien note cao, < 0: uu tien note thap
            pitch_bias = np.zeros(128)
            center = int(np.clip(64 + valence * 12 + arousal * 6, 48, 84))
            for i in range(128):
                pitch_bias[i] = -abs(i - center) * 0.05
            logits[:128] += torch.tensor(pitch_bias, dtype=torch.float32)
            # Apply scale mask
            logits_np = logits.numpy().copy()
            logits_np  = apply_scale_mask(logits_np, scale)
            logits      = torch.tensor(logits_np)
            # Sample
            probs  = F.softmax(logits / temperature, dim=0)
            next_t = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_t.unsqueeze(0)], dim=1)
            if next_t.item() < 128:  # bo REST va PAD
                notes.append(next_t.item())
    return notes if notes else [60, 64, 67]

# ── MediaPipe ─────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
holistic    = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def load_cam():
    try: return int(open(CAM_CONFIG).read().strip())
    except: return 0

def normalize_lms(lms_list, n):
    if not lms_list: return [0.0] * (n * 3)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms_list[:n]])
    center = arr[0]; scale = np.linalg.norm(arr.max(0) - arr.min(0)) + 1e-6
    return ((arr - center) / scale).flatten().tolist()

def extract_features(results):
    rh = normalize_lms(results.right_hand_landmarks.landmark  if results.right_hand_landmarks else [], 21)
    lh = normalize_lms(results.left_hand_landmarks.landmark   if results.left_hand_landmarks  else [], 21)
    po = normalize_lms(results.pose_landmarks.landmark[:33]   if results.pose_landmarks       else [], 33)
    return (rh + [0.0]*63)[:63] + (lh + [0.0]*63)[:63] + (po + [0.0]*99)[:99]

def has_motion(results):
    return (results.right_hand_landmarks is not None or
            results.left_hand_landmarks  is not None or
            results.pose_landmarks       is not None)

# ── Audio Engine ──────────────────────────────────────────────────────────
class AudioEngine:
    NOTE_DUR = 0.18  # giay moi not

    def __init__(self):
        self.fs      = fluidsynth.Synth(gain=0.8)
        self.fs.start(driver='wasapi')
        self.sfid    = self.fs.sfload(SF_PATH)
        for ch in range(2):
            self.fs.program_select(ch, self.sfid, 0, 24)
        self.lock    = threading.Lock()
        self.running = False
        self.thread  = None
        # Backing chord: channel 1
        self._backing_active = False
        self._backing_prog   = 24
        self._backing_notes  = [60, 64, 67]
        self._backing_tempo  = 100

    def play_melody(self, notes, prog, tempo, velocity=75):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        self.fs.program_select(0, self.sfid, 0, prog)
        self.running = True
        self.thread  = threading.Thread(
            target=self._melody_loop, args=(notes, tempo, velocity), daemon=True)
        self.thread.start()

    def _melody_loop(self, notes, tempo, velocity):
        beat = 60.0 / tempo
        for i, pitch in enumerate(notes):
            if not self.running: break
            vel = int(np.clip(velocity + np.random.randint(-8, 8), 40, 110))
            if i % 4 == 0: vel = min(110, vel + 15)
            with self.lock:
                self.fs.noteon(0, pitch, vel)
            time.sleep(beat * 0.75)
            with self.lock:
                self.fs.noteoff(0, pitch)
            time.sleep(beat * 0.25)

    def start_backing(self, notes, prog, tempo):
        self._backing_notes  = notes
        self._backing_prog   = prog
        self._backing_tempo  = tempo
        self._backing_active = True
        threading.Thread(target=self._backing_loop, daemon=True).start()

    def _backing_loop(self):
        self.fs.program_select(1, self.sfid, 0, self._backing_prog)
        while self._backing_active:
            beat = 60.0 / self._backing_tempo
            for n in self._backing_notes:
                self.fs.noteon(1, n, 50)
            time.sleep(beat * 2.5)
            for n in self._backing_notes:
                self.fs.noteoff(1, n)
            time.sleep(beat * 0.5)

    def update_backing(self, notes, prog, tempo):
        self._backing_notes = notes
        self._backing_tempo = tempo

    def stop_melody(self):
        self.running = False
        for n in range(128): self.fs.noteoff(0, n)

    def delete(self):
        self.running = False
        self._backing_active = False
        for n in range(128):
            self.fs.noteoff(0, n)
            self.fs.noteoff(1, n)
        self.fs.delete()

# ── UI ────────────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    ( 1, 1): "HAPPY + ENERGETIC",
    ( 1,-1): "HAPPY + CALM",
    (-1, 1): "TENSE + SAD",
    (-1,-1): "SAD + SLOW",
    ( 0, 0): "NEUTRAL",
}

def get_emotion_label(val, aro):
    vs = 1 if val > 0.2 else (-1 if val < -0.2 else 0)
    as_ = 1 if aro > 0.2 else (-1 if aro < -0.2 else 0)
    return EMOTION_LABELS.get((vs, as_), "NEUTRAL")

def draw_ui(frame, valence, arousal, inst, scale, tempo, notes, active):
    h, w = frame.shape[:2]
    color = (0,220,80) if active else (80,80,80)

    cv2.rectangle(frame,(0,0),(w-1,h-1),color,3)
    cv2.rectangle(frame,(0,0),(w,60),(0,0,0),-1)
    cv2.putText(frame,'GestuRhythm v2',(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,220,180),2)
    emo_label = get_emotion_label(valence, arousal)
    cv2.putText(frame,f'{emo_label}  |  {inst}  |  {scale}  |  {tempo}BPM',
                (10,48),cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1)

    # Chi so lon goc tren phai
    cv2.rectangle(frame, (w-220, 0), (w, 110), (0,0,0), -1)
    val_color = (0,220,80) if valence > 0.2 else ((80,80,255) if valence < -0.2 else (180,180,180))
    aro_color = (0,220,255) if arousal > 0.2 else ((255,120,0) if arousal < -0.2 else (180,180,180))
    cv2.putText(frame, f'VAL {valence:+.2f}', (w-210, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, val_color, 2)
    cv2.putText(frame, f'ARO {arousal:+.2f}', (w-210, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, aro_color, 2)
    cv2.putText(frame, get_emotion_label(valence, arousal), (w-210, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    # Emotion meter (goc duoi phai)
    cx, cy, r = w-70, h-70, 45
    cv2.circle(frame,(cx,cy),r,(40,40,40),-1)
    cv2.circle(frame,(cx,cy),r,color,2)
    cv2.line(frame,(cx-r,cy),(cx+r,cy),(80,80,80),1)
    cv2.line(frame,(cx,cy-r),(cx,cy+r),(80,80,80),1)
    px = int(cx + np.clip(valence,-.95,.95)*(r-8))
    py = int(cy - np.clip(arousal,-.95,.95)*(r-8))
    cv2.circle(frame,(px,py),8,color,-1)
    cv2.putText(frame,f'V:{valence:+.2f}',(cx-r-5,cy+r+15),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
    cv2.putText(frame,f'A:{arousal:+.2f}',(cx-r-5,cy+r+28),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)

    # Piano roll
    if notes:
        rw = (w-20) // len(notes)
        for i, n in enumerate(notes):
            bar_h = int((n - 48) / 36 * 30) + 5
            c     = (0, int(min(n/127*255,255)), int(255-n/127*255))
            cv2.rectangle(frame,(10+i*rw, h-30),(10+(i+1)*rw-2, h-30+bar_h), c, -1)
        cv2.putText(frame,'PIANO ROLL',(10,h-34),cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,160),1)

    if not active:
        cv2.putText(frame,'Dua tay vao khung hinh...',
                    (w//2-150,h//2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,100),2)

    cv2.putText(frame,'Q=thoat  SPACE=stop',(10,h-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(160,160,160),1)

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('Loading models...')
    encoder, prior, decoder = load_models()
    print(f'Encoder: {sum(p.numel() for p in encoder.parameters()):,} params')
    print(f'Prior  : {sum(p.numel() for p in prior.parameters()):,} params')

    print('Starting audio...')
    audio = AudioEngine()
    audio.start_backing([60,64,67], 24, 100)
    print('Ready!')

    cap = cv2.VideoCapture(load_cam(), cv2.CAP_MSMF)
    if not cap.isOpened():
        print('Cannot open camera'); audio.delete(); return

    frame_buf  = collections.deque(maxlen=SEQ_LEN)
    last_gen   = 0
    valence    = 0.0
    arousal    = 0.0
    notes_cur  = [60,64,67,65,69,67,64,60,62,64,65,67,69,67,65,62]
    inst       = 'Piano'
    scale      = 'pentatonic'
    tempo      = 100
    primer     = [60, 64, 67, 65]

    print('Q = quit | SPACE = stop melody')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        now   = time.time()

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        active  = has_motion(results)

        # Draw landmarks
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS)

        feat = extract_features(results)
        frame_buf.append(feat)

        # Inference moi 0.8 giay khi co chuyen dong
        if len(frame_buf) == SEQ_LEN and active and (now - last_gen) > 0.8:
            seq         = torch.tensor([list(frame_buf)], dtype=torch.float32)
            emotion_vec = encoder(seq)[0].detach()
            valence     = float(emotion_vec[0])
            arousal     = float(emotion_vec[1])

            scale, prog, inst, tempo, temp = emotion_to_config(valence, arousal)

            # Sinh melody tu Music Prior + emotion conditioning
            notes_cur = generate_notes(primer, prior, emotion_vec, scale, temp, n_notes=16, decoder=decoder)
            primer    = notes_cur[-4:]  # 4 not cuoi lam primer cho lan tiep

            # Cap nhat backing chord theo scale
            BACKING = {
                'major':    ([60,64,67], 24, tempo),
                'minor':    ([57,60,64], 48, tempo),
                'dominant': ([55,59,62], 24, tempo),
                'pentatonic':([60,64,67],0,  tempo),
            }
            audio.update_backing(*BACKING[scale])
            audio.play_melody(notes_cur, prog, tempo)
            last_gen = now

        elif not active:
            audio.stop_melody()

        draw_ui(frame, valence, arousal, inst, scale, tempo, notes_cur, active)
        cv2.imshow('GestuRhythm v2', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): audio.stop_melody()

    cap.release()
    cv2.destroyAllWindows()
    audio.delete()
    print('Done.')

if __name__ == '__main__':
    main()
