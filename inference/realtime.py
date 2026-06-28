"""
GestuRhythm - Real-time inference
Tay phai = melody (Huong C: AI sinh note theo mood)
Tay trai = nhac cu (Huong D: chord type -> instrument)
"""
import os, sys, time, collections
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import fluidsynth
import threading

os.environ['PATH'] += r';C:\FluidSynth\bin'

# ── Config ────────────────────────────────────────────────────────────────
BASE        = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH  = os.path.join(BASE, 'model', 'gesture_transformer.pt')
SF_PATH     = os.path.join(BASE, 'soundfonts', 'FluidR3_GM.sf2')
CAM_CONFIG  = os.path.join(BASE, 'camera_config.txt')
SEQ_LEN     = 30

# MIDI Program numbers (General MIDI)
INSTRUMENTS = {
    0: (24, 0,  "Acoustic Guitar"),   # chord=none
    1: (24, 0,  "Acoustic Guitar"),   # chord=major
    2: (48, 1,  "Strings"),           # chord=minor
    3: (0,  2,  "Grand Piano"),       # chord=dominant
}

# Scale snap theo chord
SCALES = {
    0: [60,62,64,65,67,69,71,72],  # C major (default)
    1: [60,62,64,65,67,69,71,72],  # C major
    2: [60,62,63,65,67,68,70,72],  # C minor
    3: [60,62,64,65,67,69,70,72],  # C dominant
}

PITCH_MIN, PITCH_MAX = 60, 72
VEL_MIN,   VEL_MAX   = 40, 127
TEMPO_MIN, TEMPO_MAX = 80, 160

# ── Model ─────────────────────────────────────────────────────────────────
class GestureTransformer(nn.Module):
    def __init__(self, input_dim=126, d_model=128, nhead=4, num_layers=3, output_dim=6):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Embedding(100, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_enc(pos)
        x = self.transformer(x)
        return self.head(x.mean(dim=1))

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']
    model = GestureTransformer(**cfg)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

def decode_output(out):
    """Chuyen tensor output ve gia tri thuc."""
    out = out.squeeze().detach().numpy()
    pitch    = float(out[0]) * (PITCH_MAX - PITCH_MIN) + PITCH_MIN
    velocity = float(out[1]) * (VEL_MAX   - VEL_MIN)   + VEL_MIN
    tempo    = float(out[2]) * (TEMPO_MAX - TEMPO_MIN)  + TEMPO_MIN
    chord    = int(round(float(out[3]) * 3))
    rp       = float(torch.sigmoid(torch.tensor(out[4]))) > 0.5
    lp       = float(torch.sigmoid(torch.tensor(out[5]))) > 0.5
    return (
        int(np.clip(pitch, PITCH_MIN, PITCH_MAX)),
        int(np.clip(velocity, VEL_MIN, VEL_MAX)),
        int(np.clip(tempo, TEMPO_MIN, TEMPO_MAX)),
        int(np.clip(chord, 0, 3)),
        bool(rp), bool(lp)
    )

def snap_to_scale(pitch, chord):
    """Snap pitch ve scale dung theo chord."""
    scale = SCALES[chord]
    return min(scale, key=lambda p: abs(p - pitch))

# ── FluidSynth ────────────────────────────────────────────────────────────
class MusicEngine:
    def __init__(self):
        self.fs   = fluidsynth.Synth(gain=0.8)
        self.fs.start(driver='wasapi')
        self.sfid = self.fs.sfload(SF_PATH)
        # Khoi tao 3 channel
        for ch in range(3):
            self.fs.program_select(ch, self.sfid, 0, 24)
        self.current_note  = {ch: None for ch in range(3)}
        self.current_chord = 0
        self.current_prog  = {0: 24, 1: 48, 2: 0}
        self.lock = threading.Lock()

    def set_instrument(self, chord):
        """Doi nhac cu theo chord type (Huong D)."""
        prog, ch, name = INSTRUMENTS[chord]
        if chord != self.current_chord:
            self.fs.program_select(ch, self.sfid, 0, prog)
            self.current_chord = chord
        return ch, name

    def play_note(self, pitch, velocity, chord):
        """Phat note tren channel tuong ung (Huong C)."""
        with self.lock:
            ch, name = self.set_instrument(chord)
            snapped = snap_to_scale(pitch, chord)
            # Tat note cu neu khac note moi
            if self.current_note[ch] is not None and self.current_note[ch] != snapped:
                self.fs.noteoff(ch, self.current_note[ch])
            # Phat note moi
            if self.current_note[ch] != snapped:
                self.fs.noteon(ch, snapped, velocity)
                self.current_note[ch] = snapped
            return snapped, ch, name

    def stop_note(self, ch=None):
        with self.lock:
            channels = range(3) if ch is None else [ch]
            for c in channels:
                if self.current_note[c] is not None:
                    self.fs.noteoff(c, self.current_note[c])
                    self.current_note[c] = None

    def stop_all(self):
        self.stop_note()

    def delete(self):
        self.stop_all()
        self.fs.delete()

# ── MediaPipe ─────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5)

def get_hand_dict(results):
    d = {}
    if not results.multi_hand_landmarks:
        return d
    for lms, info in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = info.classification[0].label
        d[label] = [(lm.x, lm.y, lm.z) for lm in lms.landmark]
    return d

def build_feature(hand_dict):
    """Tao vector 126 features tu hand_dict."""
    def flat(lms): return [v for lm in lms for v in lm]
    r = flat(hand_dict['Right']) if 'Right' in hand_dict else [0.0] * 63
    l = flat(hand_dict['Left'])  if 'Left'  in hand_dict else [0.0] * 63
    return r + l

# ── UI ────────────────────────────────────────────────────────────────────
CHORD_NAMES = {0: 'None', 1: 'Major', 2: 'Minor', 3: 'Dominant'}
NOTE_NAMES  = {60:'C4',62:'D4',63:'Eb4',64:'E4',65:'F4',67:'G4',
               68:'Ab4',69:'A4',70:'Bb4',71:'B4',72:'C5'}

def draw_ui(frame, pitch, velocity, tempo, chord, inst_name,
            rp, lp, buf_len, note_playing):
    h, w = frame.shape[:2]

    # Vung pitch ben trai
    note_names = ['C5','A4','F4','D4','C4']
    for i, name in enumerate(note_names):
        y = int(h * i / len(note_names))
        cv2.line(frame, (0, y), (45, y), (150,150,255), 1)
        cv2.putText(frame, name, (2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,255), 1)

    # Vung chord ben phai
    chord_zones = [('Major',(0,200,100)),('Major',(0,200,100)),
                   ('Dom',(0,180,255)),('Minor',(200,80,255)),('Minor',(200,80,255))]
    for i,(name,c) in enumerate(chord_zones):
        y = int(h * i / len(chord_zones))
        cv2.line(frame, (w-55,y),(w,y), c, 1)
        cv2.putText(frame, name, (w-52,y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1)

    # Duong vung ly tuong
    cv2.line(frame, (55,int(h*0.2)),(w-60,int(h*0.2)),(0,255,200),1)
    cv2.line(frame, (55,int(h*0.8)),(w-60,int(h*0.8)),(0,255,200),1)

    # Header
    cv2.rectangle(frame, (0,0),(w,58),(0,0,0),-1)
    cv2.putText(frame, 'GestuRhythm - Real-time', (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)
    cv2.putText(frame, f'Instrument: {inst_name}  |  Chord: {CHORD_NAMES[chord]}',
                (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Thong so
    rc = (0,220,80) if rp else (80,80,80)
    lc = (0,180,255) if lp else (80,80,80)
    cv2.putText(frame, f'R[{"ON" if rp else "OFF"}] P:{NOTE_NAMES.get(pitch,pitch)} V:{velocity} T:{tempo}',
                (w-280,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 2)
    cv2.putText(frame, f'L[{"ON" if lp else "OFF"}] Chord:{CHORD_NAMES[chord]}',
                (w-280,102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)

    # Note dang phat (nhip nhay)
    if note_playing:
        alpha = int(180 + 75 * abs(np.sin(time.time() * 6)))
        color = (0, alpha, 100)
        cv2.putText(frame, f'NOTE: {NOTE_NAMES.get(note_playing, note_playing)}',
                    (w//2-60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Buffer bar
    bw = int((w-10) * buf_len / SEQ_LEN)
    cv2.rectangle(frame,(5,h-25),(w-5,h-5),(40,40,40),-1)
    cv2.rectangle(frame,(5,h-25),(5+bw,h-5),(0,140,255),-1)
    cv2.putText(frame,'Q=thoat  SPACE=stop note',(10,h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200,200,200),1)

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('Dang tai model...')
    model = load_model()
    print('Model OK')

    print('Khoi dong FluidSynth...')
    engine = MusicEngine()
    print('FluidSynth OK')

    try:
        cam_idx = int(open(CAM_CONFIG).read().strip())
    except Exception:
        cam_idx = 0

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f'Khong mo duoc camera {cam_idx}')
        engine.delete()
        return

    frame_buffer = collections.deque(maxlen=SEQ_LEN)
    last_pitch   = 60
    note_playing = None
    pitch = velocity = tempo = 60
    chord = 0
    rp = lp = False
    inst_name = 'Acoustic Guitar'

    print('Bat dau! Dua tay vao khung hinh.')
    print('Q = thoat | SPACE = stop note')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_dict = get_hand_dict(results)

        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        feat = build_feature(hand_dict)
        frame_buffer.append(feat)

        if len(frame_buffer) == SEQ_LEN:
            seq = torch.tensor([list(frame_buffer)], dtype=torch.float32)
            with torch.no_grad():
                out = model(seq)
            pitch, velocity, tempo, chord, rp, lp = decode_output(out)

            if rp:
                snapped, ch, inst_name = engine.play_note(pitch, velocity, chord)
                note_playing = snapped
                last_pitch   = snapped
            else:
                engine.stop_all()
                note_playing = None

        draw_ui(frame, pitch, velocity, tempo, chord, inst_name,
                rp, lp, len(frame_buffer), note_playing)

        cv2.imshow('GestuRhythm', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            engine.stop_all()
            note_playing = None

    cap.release()
    cv2.destroyAllWindows()
    engine.delete()
    print('Thoat.')

if __name__ == '__main__':
    main()
