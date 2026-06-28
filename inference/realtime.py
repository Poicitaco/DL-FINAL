"""
GestuRhythm - Real-time inference voi Seq2Seq model
Tay phai = melody (model sinh 16 not)
Tay trai = chord (Major/Minor/Dominant)
"""
import os, sys, time, collections
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import fluidsynth
import threading

os.environ['PATH'] += r';C:\tools\fluidsynth\bin'

BASE       = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(BASE, 'model', 'gesture_seq2seq.pt')
SF_PATH    = os.path.join(BASE, 'soundfonts', 'FluidR3_GM.sf2')
CAM_CONFIG = os.path.join(BASE, 'camera_config.txt')
SEQ_LEN    = 30
NOTE_OUT   = 16
NOTE_DIM   = 3

PITCH_MIN, PITCH_MAX = 60, 72
VEL_MIN,   VEL_MAX   = 40, 127
TEMPO_MIN, TEMPO_MAX = 80, 160

CHORD_SCALES = {
    0: [60,62,64,65,67,69,71,72],
    1: [60,62,64,65,67,69,71,72],
    2: [60,62,63,65,67,68,70,72],
    3: [60,62,64,65,67,69,70,72],
}
INSTRUMENTS = {0:(24,"Guitar"), 1:(24,"Guitar"), 2:(48,"Strings"), 3:(0,"Piano")}
CHORD_NAMES = {0:"None", 1:"Major", 2:"Minor", 3:"Dominant"}
NOTE_NAMES  = {60:"C4",62:"D4",63:"Eb4",64:"E4",65:"F4",67:"G4",
               68:"Ab4",69:"A4",70:"Bb4",71:"B4",72:"C5"}

# ── Model ─────────────────────────────────────────────────────────────────
class GestureEncoder(nn.Module):
    def __init__(self, input_dim=126, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Embedding(100, d_model)
        enc_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        x    = self.embed(x) + self.pos_enc(pos)
        return self.transformer(x, mask=mask)

class MusicDecoder(nn.Module):
    def __init__(self, note_dim=NOTE_DIM, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.note_embed = nn.Linear(note_dim, d_model)
        self.pos_enc    = nn.Embedding(100, d_model)
        dec_layer       = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(d_model, note_dim)

    def forward(self, tgt, memory):
        B, T, _ = tgt.shape
        mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        pos  = torch.arange(T, device=tgt.device).unsqueeze(0)
        tgt  = self.note_embed(tgt) + self.pos_enc(pos)
        return self.out_proj(self.transformer(tgt, memory, tgt_mask=mask))

class GestureSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GestureEncoder()
        self.decoder = MusicDecoder()

    def generate(self, src, seq_len=NOTE_OUT):
        self.eval()
        with torch.no_grad():
            memory    = self.encoder(src)
            generated = torch.zeros(1, 1, NOTE_DIM, device=src.device)
            for _ in range(seq_len):
                out       = self.decoder(generated, memory)
                generated = torch.cat([generated, out[:,-1:,:]], dim=1)
            return generated[:, 1:, :]

def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = GestureSeq2Seq()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

# ── FluidSynth ────────────────────────────────────────────────────────────
class MusicEngine:
    def __init__(self):
        self.fs   = fluidsynth.Synth(gain=0.8)
        self.fs.start(driver='wasapi')
        self.sfid = self.fs.sfload(SF_PATH)
        for ch in range(4):
            self.fs.program_select(ch, self.sfid, 0, 24)
        self.playing_notes = {}
        self.lock = threading.Lock()
        self.seq_thread = None
        self.running    = False

    def play_sequence(self, notes, chord, tempo):
        """Phat chuoi 16 not tren background thread."""
        self.running = False
        if self.seq_thread and self.seq_thread.is_alive():
            self.seq_thread.join(timeout=0.5)

        prog, inst = INSTRUMENTS[chord]
        self.fs.program_select(0, self.sfid, 0, prog)
        self.running    = True
        self.seq_thread = threading.Thread(
            target=self._play_loop, args=(notes, chord, tempo), daemon=True)
        self.seq_thread.start()
        return inst

    def _play_loop(self, notes, chord, tempo):
        scale = CHORD_SCALES[chord]
        beat  = 60.0 / max(tempo, 80)
        for i, note in enumerate(notes):
            if not self.running: break
            pitch    = int(np.clip(float(note[0]) * (PITCH_MAX-PITCH_MIN) + PITCH_MIN, PITCH_MIN, PITCH_MAX))
            velocity = int(np.clip(float(note[1]) * (VEL_MAX-VEL_MIN) + VEL_MIN, VEL_MIN, VEL_MAX))
            duration = float(np.clip(float(note[2]) * beat * 1.5, 0.08, beat * 1.5))
            pitch    = min(scale, key=lambda s: abs(s - pitch))
            # Them accent: nhat manh hon moi 4 not
            if i % 4 == 0:
                velocity = min(127, velocity + 20)
            with self.lock:
                if 0 in self.playing_notes:
                    self.fs.noteoff(0, self.playing_notes[0])
                self.fs.noteon(0, pitch, velocity)
                self.playing_notes[0] = pitch
            # Note on 80% duration, off 20% -> co articulation
            time.sleep(duration * 0.75)
            with self.lock:
                self.fs.noteoff(0, pitch)
                if 0 in self.playing_notes:
                    del self.playing_notes[0]
            time.sleep(duration * 0.25)
        with self.lock:
            if 0 in self.playing_notes:
                self.fs.noteoff(0, self.playing_notes[0])
                del self.playing_notes[0]

    def start_backing(self, chord):
        """Phat chord loop nen tren channel 1."""
        self.stop_backing()
        BACKING = {
            0: ([60,64,67],    90,  0),   # C major, guitar
            1: ([60,64,67],    90,  0),   # C major, guitar
            2: ([57,60,64],    80, 48),   # Am, strings
            3: ([55,59,62,67], 100, 0),   # G dominant, guitar
        }
        notes, tempo, prog = BACKING[chord]
        self.fs.program_select(1, self.sfid, 0, prog)
        self._backing_notes  = notes
        self._backing_tempo  = tempo
        self._backing_active = True
        self._backing_chord  = chord
        t = threading.Thread(target=self._backing_loop, daemon=True)
        t.start()

    def _backing_loop(self):
        while getattr(self, '_backing_active', False):
            beat = 60.0 / self._backing_tempo
            # Phat chord (tat ca not cung luc)
            for n in self._backing_notes:
                self.fs.noteon(1, n, 55)
            time.sleep(beat * 3)
            for n in self._backing_notes:
                self.fs.noteoff(1, n)
            time.sleep(beat * 0.2)
            # Doi chord neu thay doi
            if getattr(self, '_backing_chord', 0) != getattr(self, '_pending_chord', self._backing_chord):
                self._backing_chord = self._pending_chord

    def stop_backing(self):
        self._backing_active = False
        for n in range(128):
            self.fs.noteoff(1, n)

    def set_backing_chord(self, chord):
        self._pending_chord = chord

    def stop_all(self):
        self.running = False
        with self.lock:
            for ch, note in self.playing_notes.items():
                self.fs.noteoff(ch, note)
            self.playing_notes.clear()

    def delete(self):
        self.stop_all()
        self.stop_backing()
        self.fs.delete()

# ── MediaPipe ─────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5)

def load_cam():
    try: return int(open(CAM_CONFIG).read().strip())
    except: return 0

def get_hand_dict(results):
    d = {}
    if not results.multi_hand_landmarks:
        return d
    for lms, info in zip(results.multi_hand_landmarks, results.multi_handedness):
        d[info.classification[0].label] = [(lm.x,lm.y,lm.z) for lm in lms.landmark]
    return d

def normalize_landmarks(lms):
    wrist = np.array(lms[0])
    scale = np.linalg.norm(np.array(lms[9]) - wrist) + 1e-6
    return [((x-wrist[0])/scale,(y-wrist[1])/scale,(z-wrist[2])/scale) for x,y,z in lms]

def build_feature(hand_dict):
    ZEROS = [0.0] * 63
    r = [v for lm in normalize_landmarks(hand_dict['Right']) for v in lm] if 'Right' in hand_dict else ZEROS
    l = [v for lm in normalize_landmarks(hand_dict['Left'])  for v in lm] if 'Left'  in hand_dict else ZEROS
    return r + l

def get_chord(hand_dict):
    CHORD_MAP = [1,1,3,2,2]
    if 'Left' not in hand_dict: return 0
    wy  = hand_dict['Left'][0][1]
    row = min(int(wy * len(CHORD_MAP)), len(CHORD_MAP)-1)
    return CHORD_MAP[row]

# ── UI ────────────────────────────────────────────────────────────────────
def draw_ui(frame, notes_playing, chord, inst, rp, lp, buf_len, tempo, history):
    h, w = frame.shape[:2]
    color = (0,200,255) if rp else (100,100,100)
    cv2.rectangle(frame, (0,0),(w-1,h-1), color, 3)

    cv2.rectangle(frame,(0,0),(w,55),(0,0,0),-1)
    cv2.putText(frame, f'GestuRhythm  |  {inst}  |  {CHORD_NAMES[chord]}  |  {tempo}BPM',
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f'TAY PHAI[{"ON" if rp else "OFF"}]   TAY TRAI[{"ON" if lp else "OFF"}]',
                (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Piano roll - hien thi 16 not dang phat
    if notes_playing is not None:
        roll_x, roll_y = 10, h - 80
        roll_w = w - 20
        note_w = roll_w // NOTE_OUT
        for i, note in enumerate(notes_playing):
            p = int(note[0] * (PITCH_MAX-PITCH_MIN) + PITCH_MIN)
            v = note[1]
            bar_h = int(v * 30) + 5
            py    = int((1 - (p-PITCH_MIN)/(PITCH_MAX-PITCH_MIN)) * 25) + roll_y
            c     = (0, int(v*255), int((1-v)*255))
            cv2.rectangle(frame, (roll_x + i*note_w, py),
                          (roll_x + (i+1)*note_w - 2, py+bar_h), c, -1)
        cv2.putText(frame, 'PIANO ROLL', (10, roll_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    # Buffer
    bw = int((w-10) * buf_len / SEQ_LEN)
    cv2.rectangle(frame,(5,h-25),(w-5,h-5),(40,40,40),-1)
    cv2.rectangle(frame,(5,h-25),(5+bw,h-5),(0,140,255),-1)
    cv2.putText(frame,'Q=thoat  SPACE=stop',(10,h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200,200,200),1)

    # History panel - hien thi 8 thay doi gan nhat
    panel_x = 5
    cv2.rectangle(frame, (panel_x, 58), (220, 58 + 8*22 + 5), (0,0,0), -1)
    cv2.putText(frame, 'LICH SU THAY DOI:', (panel_x+3, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
    for i, (ts, event, color) in enumerate(list(history)[-8:]):
        y = 90 + i * 20
        cv2.putText(frame, event, (panel_x+3, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Chord zones ben phai
    zone_h = h // 5
    for i, (name, c) in enumerate([("MAJOR",(0,200,100)),("MAJOR",(0,200,100)),
                                    ("DOM",(0,180,255)),("MINOR",(200,80,255)),("MINOR",(200,80,255))]):
        y = i * zone_h
        cv2.line(frame,(w-55,y),(w,y),c,1)
        cv2.putText(frame,name,(w-52,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.35,c,1)

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print('Loading model...')
    model = load_model()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    print('Starting FluidSynth...')
    engine = MusicEngine()
    print('FluidSynth OK')

    cap = cv2.VideoCapture(load_cam(), cv2.CAP_MSMF)
    if not cap.isOpened():
        print('Cannot open camera'); engine.delete(); return

    frame_buffer  = collections.deque(maxlen=SEQ_LEN)
    history       = collections.deque(maxlen=8)
    notes_playing = None
    inst_name     = 'Guitar'
    chord         = 0
    tempo         = 120
    rp = lp       = False
    last_gen      = 0
    last_chord    = -1
    last_rp       = False

    print('Ready! Move your hands.')
    print('Q = quit | SPACE = stop notes')

    engine.start_backing(0)  # bat dau backing track

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_dict = get_hand_dict(results)

        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        rp    = 'Right' in hand_dict
        lp    = 'Left'  in hand_dict
        chord = get_chord(hand_dict)
        engine.set_backing_chord(chord)

        # Log thay doi
        if rp != last_rp:
            if rp: history.append((time.time(), '+ Tay phai bat dau', (0,220,80)))
            else:  history.append((time.time(), '- Tay phai dung',    (100,100,100)))
            last_rp = rp
        if chord != last_chord and last_chord >= 0:
            history.append((time.time(), f'Chord: {CHORD_NAMES[last_chord]}->{CHORD_NAMES[chord]}', (0,180,255)))
            last_chord = chord
        elif last_chord < 0:
            last_chord = chord

        feat = build_feature(hand_dict)
        frame_buffer.append(feat)

        now = time.time()
        # Sinh chuoi not moi moi 1 giay neu co tay phai
        if len(frame_buffer) == SEQ_LEN and rp and (now - last_gen) > 0.5:
            seq    = torch.tensor([list(frame_buffer)], dtype=torch.float32)
            notes  = model.generate(seq)[0].numpy()  # (16, 3)
            # Tinh tempo tu velocity trung binh
            tempo  = int(np.clip(float(notes[:,1].mean()) * (TEMPO_MAX-TEMPO_MIN) + TEMPO_MIN,
                                 TEMPO_MIN, TEMPO_MAX))
            inst_name = engine.play_sequence(notes, chord, tempo)
            notes_playing = notes
            last_gen      = now
            # Log note moi
            avg_p = int(float(notes[:,0].mean()) * (PITCH_MAX-PITCH_MIN) + PITCH_MIN)
            history.append((now, f'Note: {NOTE_NAMES.get(avg_p,avg_p)} {tempo}BPM {inst_name}', (255,200,0)))
        elif not rp:
            engine.stop_all()
            notes_playing = None

        draw_ui(frame, notes_playing, chord, inst_name, rp, lp, len(frame_buffer), tempo, history)
        cv2.imshow('GestuRhythm', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): engine.stop_all(); notes_playing = None

    cap.release()
    cv2.destroyAllWindows()
    engine.stop_backing()
    engine.delete()
    print('Done.')

if __name__ == '__main__':
    main()
