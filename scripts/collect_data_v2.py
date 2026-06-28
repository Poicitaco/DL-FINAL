"""
GestuRhythm v2 - Thu thap data cam xuc
6 che do cam xuc: HAPPY_HIGH, HAPPY_LOW, SAD_HIGH, SAD_LOW, NEUTRAL, NONE
Input: MediaPipe Holistic (21 hand + 33 pose landmarks)
Output: gesture_data_v2.csv
"""
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import threading

os.environ['PATH'] += r';C:\tools\fluidsynth\bin'

# ── Config ────────────────────────────────────────────────────────────────
CAM_CONFIG  = os.path.join(os.path.dirname(__file__), '..', 'camera_config.txt')
OUTPUT_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'gesture_data_v2.csv')
FRAMES_PER_SAMPLE = 30
SAMPLES_PER_MODE  = 150

# Vector cam xuc [valence, arousal] cho moi che do
# valence: -1=buon, +1=vui | arousal: -1=thu gian, +1=nang luong cao
MODES = [
    {
        "id":      0,
        "name":    "HAPPY_HIGH",
        "desc":    "Vui + Nang luong cao | Tay NHANH, RONG, LEN CAO",
        "emotion": [1.0,  1.0],
        "color":   (0, 220, 80),
        "music":   (24, [60,64,67,72], 150),  # Guitar, C major, nhanh
        "guide":   "Di chuyen tay NHANH va RONG, nhu dang nhay mua",
    },
    {
        "id":      1,
        "name":    "HAPPY_LOW",
        "desc":    "Vui + Nhe nhang | Tay CHAM, TRON, NHE",
        "emotion": [1.0, -0.5],
        "color":   (0, 180, 255),
        "music":   (24, [60,64,67,69], 85),   # Guitar, C major, cham
        "guide":   "Di chuyen tay CHAM va NGANG, nhu dang lau kinh",
    },
    {
        "id":      2,
        "name":    "SAD_HIGH",
        "desc":    "Buon + Cang thang | Tay RUNG, GAP, XUONG THAP",
        "emotion": [-1.0, 0.8],
        "color":   (80, 0, 255),
        "music":   (48, [57,60,63], 110),     # Strings, Am, vua
        "guide":   "Tay xuong thap, di chuyen GAP va CO RUT lai",
    },
    {
        "id":      3,
        "name":    "SAD_LOW",
        "desc":    "Buon + Cham | Tay XUONG THAP, rat CHAM",
        "emotion": [-1.0, -1.0],
        "color":   (150, 0, 200),
        "music":   (48, [57,60,64], 65),      # Strings, Am, rat cham
        "guide":   "Tay xuong thap nhat, di chuyen RAT CHAM va NANG NE",
    },
    {
        "id":      4,
        "name":    "NEUTRAL",
        "desc":    "Trung tinh | Tay di chuyen DEU, o GIUA man hinh",
        "emotion": [0.0,  0.0],
        "color":   (180, 180, 0),
        "music":   (0,  [60,64,67], 100),     # Piano, C major, vua
        "guide":   "Tay o giua man hinh, di chuyen DEU DANG, khong qua nhanh hay cham",
    },
    {
        "id":      5,
        "name":    "NONE",
        "desc":    "Khong co tay | De tay xuong, ra ngoai khung hinh",
        "emotion": [0.0,  0.0],
        "color":   (80, 80, 80),
        "music":   (None, [], 0),
        "guide":   "Buong tay xuong, KHONG co tay trong khung hinh",
    },
]

# ── MediaPipe Holistic ─────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
holistic    = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def load_cam():
    try: return int(open(CAM_CONFIG).read().strip())
    except: return 0

def normalize_lms(lms_list):
    """Normalize landmarks ve relative coords."""
    if not lms_list:
        return [0.0] * (len(lms_list[0]) * 3 if lms_list else 63)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms_list])
    center = arr[0]
    scale  = np.linalg.norm(arr.max(0) - arr.min(0)) + 1e-6
    arr    = (arr - center) / scale
    return arr.flatten().tolist()

def extract_features(results):
    """Trich xuat 21 hand + 33 pose landmarks = 162 features."""
    # Hand phai (63), hand trai (63), pose (99)
    rh = normalize_lms(results.right_hand_landmarks.landmark  if results.right_hand_landmarks  else []) or [0.0]*63
    lh = normalize_lms(results.left_hand_landmarks.landmark   if results.left_hand_landmarks   else []) or [0.0]*63
    po = normalize_lms(results.pose_landmarks.landmark[:33]   if results.pose_landmarks        else []) or [0.0]*99
    # Pad neu thieu
    rh = (rh + [0.0]*63)[:63]
    lh = (lh + [0.0]*63)[:63]
    po = (po + [0.0]*99)[:99]
    return rh + lh + po  # 225 features

def has_hand(results):
    return results.right_hand_landmarks is not None or results.left_hand_landmarks is not None

# ── Music player cho nen ──────────────────────────────────────────────────
class BgMusic:
    def __init__(self):
        self.fs = None
        self.running = False
        try:
            import fluidsynth
            sf = os.path.join(os.path.dirname(__file__), '..', 'soundfonts', 'FluidR3_GM.sf2')
            self.fs   = fluidsynth.Synth(gain=0.5)
            self.fs.start(driver='wasapi')
            self.sfid = self.fs.sfload(sf)
            self.fs.program_select(0, self.sfid, 0, 24)
        except: pass

    def play(self, prog, notes, tempo):
        self.stop()
        if not self.fs or not notes: return
        self.fs.program_select(0, self.sfid, 0, prog)
        self.running = True
        threading.Thread(target=self._loop, args=(notes, tempo), daemon=True).start()

    def _loop(self, notes, tempo):
        beat = 60.0 / tempo
        while self.running:
            for n in notes:
                if not self.running: break
                self.fs.noteon(0, n, 65)
                time.sleep(beat * 0.85)
                self.fs.noteoff(0, n)
                time.sleep(beat * 0.15)

    def stop(self):
        self.running = False
        if self.fs:
            for n in range(128): self.fs.noteoff(0, n)

    def delete(self):
        self.stop()
        if self.fs: self.fs.delete()

# ── UI ────────────────────────────────────────────────────────────────────
def draw_emotion_meter(frame, valence, arousal, color):
    """Ve dong ho cam xuc 2 truc."""
    h, w = frame.shape[:2]
    cx, cy = w - 80, h - 80
    r = 50
    cv2.circle(frame, (cx, cy), r, (50,50,50), -1)
    cv2.circle(frame, (cx, cy), r, color, 2)
    # Truc
    cv2.line(frame, (cx-r, cy), (cx+r, cy), (80,80,80), 1)
    cv2.line(frame, (cx, cy-r), (cx, cy+r), (80,80,80), 1)
    # Diem cam xuc
    px = int(cx + valence * (r-10))
    py = int(cy - arousal * (r-10))
    cv2.circle(frame, (px, py), 8, color, -1)
    cv2.putText(frame, "V+", (cx+r-15, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120,120,120), 1)
    cv2.putText(frame, "A+", (cx-8, cy-r+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120,120,120), 1)

def draw_hud(frame, mode, mode_idx, sample_count, buf_len, has_h):
    h, w = frame.shape[:2]
    color = mode["color"]
    val, aro = mode["emotion"]

    cv2.rectangle(frame, (0,0),(w-1,h-1), color, 3)
    cv2.rectangle(frame, (0,0),(w,60),(0,0,0),-1)
    cv2.putText(frame, f"[{mode_idx+1}/{len(MODES)}] {mode['name']}",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, mode["desc"],
                (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # Huong dan
    cv2.rectangle(frame, (0,62),(w,100),(20,20,20),-1)
    cv2.putText(frame, f"HUONG DAN: {mode['guide']}",
                (10,84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,220,0), 1)

    # Emotion values
    cv2.putText(frame, f"Valence:{val:+.1f}  Arousal:{aro:+.1f}",
                (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Hand status
    hc = (0,220,80) if has_h else (80,80,80)
    cv2.putText(frame, f"TAY: {'DETECT' if has_h else 'KHONG THAY'}",
                (10,145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hc, 2)

    draw_emotion_meter(frame, val, aro, color)

    # Duong vung ly tuong
    cv2.line(frame,(0,int(h*0.2)),(w,int(h*0.2)),(0,255,200),1)
    cv2.line(frame,(0,int(h*0.8)),(w,int(h*0.8)),(0,255,200),1)

    # Buffer
    bw = int((w-10)*buf_len/FRAMES_PER_SAMPLE)
    cv2.rectangle(frame,(5,h-45),(w-5,h-28),(40,40,40),-1)
    cv2.rectangle(frame,(5,h-45),(5+bw,h-28),(0,140,255),-1)
    cv2.putText(frame,f"Buffer {buf_len}/{FRAMES_PER_SAMPLE}",
                (10,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,140,255),1)

    # Progress
    pw = int((w-10)*sample_count/SAMPLES_PER_MODE)
    cv2.rectangle(frame,(5,h-25),(w-5,h-5),(40,40,40),-1)
    cv2.rectangle(frame,(5,h-25),(5+pw,h-5),(0,200,80),-1)
    cv2.putText(frame,f"Samples: {sample_count}/{SAMPLES_PER_MODE}  |  SPACE=pause  Q=quit",
                (10,h-28),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

def show_intro(cap, mode, mode_idx):
    while True:
        ret, frame = cap.read()
        if not ret: return False
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ov = np.zeros_like(frame)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        c = mode["color"]
        cv2.putText(frame, f"CHE DO {mode_idx+1}/{len(MODES)}",
                    (w//2-110, h//2-90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        cv2.putText(frame, mode["name"],
                    (w//2-150, h//2-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
        cv2.putText(frame, mode["desc"],
                    (w//2-230, h//2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 2)
        cv2.putText(frame, f"HUONG DAN: {mode['guide']}",
                    (w//2-230, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,220,0), 1)
        cv2.putText(frame, "Nhan SPACE de bat dau",
                    (w//2-160, h//2+100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,180), 2)
        cv2.imshow("GestuRhythm v2 - Thu Data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            for cnt in [3,2,1]:
                ret2, f2 = cap.read()
                if ret2:
                    f2 = cv2.flip(f2, 1)
                    cv2.putText(f2, str(cnt), (f2.shape[1]//2-40, f2.shape[0]//2+40),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, c, 8)
                    cv2.imshow("GestuRhythm v2 - Thu Data", f2)
                    cv2.waitKey(1000)
            return True
        if key == ord('q'): return False

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    cam_idx = load_cam()
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"Khong mo duoc camera {cam_idx}"); return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)

    # Header: timestamp + emotion[2] + mode_id + person_id + 225 features
    feat_cols = ([f"rh_{i}_{a}" for i in range(21) for a in "xyz"] +
                 [f"lh_{i}_{a}" for i in range(21) for a in "xyz"] +
                 [f"po_{i}_{a}" for i in range(33) for a in "xyz"])
    headers = ['timestamp','valence','arousal','mode_id','person_id'] + feat_cols

    # Hoi ten nguoi thu
    person_id = input("Nhap ten/ma nguoi thu (vd: A, B, C): ").strip() or "unknown"

    # Chon che do
    print("\nCac che do:")
    for i, m in enumerate(MODES):
        print(f"  [{i+1}] {m['name']} — {m['desc']}")
    print("  [0] Thu tat ca\n")
    choice = input("Chon (0=tat ca, vd: 1,3,5): ").strip()
    if choice == '0' or choice == '':
        selected = list(range(len(MODES)))
    else:
        try:
            selected = [int(x)-1 for x in choice.split(',') if 0 < int(x) <= len(MODES)]
        except: selected = list(range(len(MODES)))

    music = BgMusic()
    total = 0

    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(headers)

        for idx in selected:
            mode = MODES[idx]
            if not show_intro(cap, mode, idx): break

            # Bat nhac nen
            prog, notes, tempo = mode["music"]
            if prog is not None and notes:
                music.play(prog, notes, tempo)

            sample_count = 0
            frame_buffer = []
            paused = False

            while sample_count < SAMPLES_PER_MODE:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                now = time.time()

                if not paused:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)

                    # Ve landmarks
                    if results.right_hand_landmarks:
                        mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                               mp_holistic.HAND_CONNECTIONS)
                    if results.left_hand_landmarks:
                        mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                               mp_holistic.HAND_CONNECTIONS)
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                               mp_holistic.POSE_CONNECTIONS)

                    feats   = extract_features(results)
                    has_h   = has_hand(results)
                    val, ar = mode["emotion"]

                    frame_buffer.append([now, val, ar, mode["id"], person_id] + feats)

                    if len(frame_buffer) >= FRAMES_PER_SAMPLE:
                        # Che do NONE: can khong co tay
                        if mode["id"] == 5:
                            valid = sum(1 for r in frame_buffer
                                        if r[5+63+63] == 0.0 and r[5] == 0.0) >= 25
                        else:
                            # Co tay: it nhat 10/30 frame co bat ki feature nao khac 0
                            valid = sum(1 for r in frame_buffer
                                        if any(v != 0.0 for v in r[5:5+63]) or
                                           any(v != 0.0 for v in r[5+63:5+126])) >= 10
                        if valid:
                            for row in frame_buffer: writer.writerow(row)
                            sample_count += 1
                            if sample_count % 20 == 0:
                                print(f"  [{mode['name']}] {sample_count}/{SAMPLES_PER_MODE}")
                        frame_buffer.clear()
                else:
                    has_h = False

                draw_hud(frame, mode, idx, sample_count, len(frame_buffer), has_h if not paused else False)
                if paused:
                    h2, w2 = frame.shape[:2]
                    cv2.putText(frame, "TAM DUNG - SPACE de tiep tuc",
                                (w2//2-200, h2//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.imshow("GestuRhythm v2 - Thu Data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): music.stop(); total += sample_count; break
                elif key == ord(' '): paused = not paused

            music.stop()
            total += sample_count
            print(f"  [{mode['name']}] Xong: {sample_count} samples")

    cap.release()
    cv2.destroyAllWindows()
    music.delete()
    print(f"\nTong: {total} samples -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
