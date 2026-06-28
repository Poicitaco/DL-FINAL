"""
Thu thập data landmarks 2 tay - mỗi tay có vai trò âm nhạc riêng.
Tay phải = Melody (pitch, velocity)
Tay trái = Chord (major/minor/dominant/none)

Input:  126 features = tay phải (63) + tay trái (63), thiếu tay → zeros
Output: pitch, velocity, tempo, chord_type
"""
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ── Cấu hình ──────────────────────────────────────────────────────────────
CAMERA_CONFIG     = os.path.join(os.path.dirname(__file__), '..', 'camera_config.txt')
OUTPUT_CSV        = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'gesture_data.csv')
FRAMES_PER_SAMPLE = 30
SAMPLES_PER_MODE  = 150

PITCH_MAP = [72, 69, 65, 62, 60]  # C5 A4 F4 D4 C4

# chord theo vị trí tay trái: trên = major(1), giữa = dominant(3), dưới = minor(2)
CHORD_MAP = [1, 1, 3, 2, 2]  # 0=none, 1=major, 2=minor, 3=dominant

MODES = [
    # ── Tay phai ──────────────────────────────────────────────────────────
    {
        "name":   "1. TAY PHAI - LEN XUONG",
        "desc":   "Tay PHAI len xuong cham | Tay TRAI de sau lung",
        "arrow":  "RIGHT_UD",
        "color":  (0, 200, 255),
        "active": "right",
    },
    {
        "name":   "2. TAY PHAI - NHANH CHAM",
        "desc":   "Tay PHAI vay nhanh roi cham xen ke | Tay TRAI de sau lung",
        "arrow":  "RIGHT_FAST",
        "color":  (0, 100, 255),
        "active": "right",
    },
    {
        "name":   "3. TAY PHAI - VONG TRON",
        "desc":   "Tay PHAI ve vong tron lien tuc | Tay TRAI de sau lung",
        "arrow":  "RIGHT_CIRCLE",
        "color":  (0, 60, 255),
        "active": "right",
    },
    {
        "name":   "4. TAY PHAI - DAY RA KÉO VAO",
        "desc":   "Tay PHAI day ra xa camera roi keo lai | Nhu dang chay vao tuong",
        "arrow":  "RIGHT_ZD",
        "color":  (100, 80, 255),
        "active": "right",
    },
    # ── Tay trai ──────────────────────────────────────────────────────────
    {
        "name":   "5. TAY TRAI - LEN XUONG",
        "desc":   "Tay TRAI len xuong cham | Tay PHAI de sau lung",
        "arrow":  "LEFT_UD",
        "color":  (255, 150, 0),
        "active": "left",
    },
    {
        "name":   "6. TAY TRAI - 3 VUNG CAO GIUA THAP",
        "desc":   "Tay TRAI: Giữ o vung TREN 2s, GIUA 2s, DUOI 2s, lap lai",
        "arrow":  "LEFT_ZONES",
        "color":  (200, 80, 255),
        "active": "left",
    },
    # ── 2 tay ─────────────────────────────────────────────────────────────
    {
        "name":   "7. 2 TAY - MO RONG THU HEP",
        "desc":   "2 tay tu giua day ra 2 ben, roi keo vao giua, lap lai",
        "arrow":  "BOTH_EXPAND",
        "color":  (0, 220, 120),
        "active": "both",
    },
    {
        "name":   "8. 2 TAY - CUNG LEN XUONG",
        "desc":   "Ca 2 tay cung di chuyen len xuong dong thoi",
        "arrow":  "BOTH_SAME",
        "color":  (0, 200, 200),
        "active": "both",
    },
    {
        "name":   "9. TAY PHAI NHANH + TRAI GIU",
        "desc":   "Tay TRAI giu o GIUA man hinh | Tay PHAI vay nhanh tu do",
        "arrow":  "RIGHT_FAST_LEFT_HOLD",
        "color":  (255, 200, 0),
        "active": "both",
    },
    # ── Dong tac dac biet ─────────────────────────────────────────────────
    {
        "name":   "10. TAY PHAI - ZIGZAG",
        "desc":   "Tay PHAI di chuyen zic-zac: len-xuong-len nhanh lien tuc",
        "arrow":  "RIGHT_ZZ",
        "color":  (255, 80, 80),
        "active": "right",
    },
    # ── Khong tay ─────────────────────────────────────────────────────────
    {
        "name":   "11. KHONG CO TAY",
        "desc":   "De ca 2 tay xuong, ngoi yen, khong co tay trong khung hinh",
        "arrow":  "NONE",
        "color":  (100, 100, 100),
        "active": "none",
    },
]

# ── MediaPipe ─────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5)

def load_camera_index():
    try:
        with open(CAMERA_CONFIG) as f:
            return int(f.read().strip())
    except Exception:
        return 0

def normalize_landmarks(lms):
    """Normalize ve relative coords - wrist = goc (0,0,0), scale theo wrist->lm9."""
    wrist = np.array(lms[0])
    scale = np.linalg.norm(np.array(lms[9]) - wrist) + 1e-6
    return [((x - wrist[0])/scale, (y - wrist[1])/scale, (z - wrist[2])/scale)
            for x, y, z in lms]

def get_hand_dict(results):
    """Trả về dict {'Right': landmarks, 'Left': landmarks} hoặc rỗng."""
    hand_dict = {}
    if not results.multi_hand_landmarks:
        return hand_dict
    for hand_lms, hand_info in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
        label = hand_info.classification[0].label  # 'Left' hoặc 'Right'
        hand_dict[label] = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
    return hand_dict

def auto_label(hand_dict):
    """Gán nhãn từ vị trí tay phải (melody) và tay trái (chord)."""
    ZEROS = [0.0] * 63

    # ── Tay phải → pitch, velocity, tempo ──
    if 'Right' in hand_dict:
        lms   = hand_dict['Right']
        wx, wy = lms[0][0], lms[0][1]
        # Dung dau ngon giua (lm12) de tinh pitch - diem cao nhat cua ban tay
        finger_y = lms[12][1]
        row   = min(int(finger_y * len(PITCH_MAP)), len(PITCH_MAP) - 1)
        pitch = PITCH_MAP[row]
        tempo = int(np.clip(80 + wx * 80, 80, 160))
        flat_right = [v for lm in normalize_landmarks(lms) for v in lm]
        right_present = 1
    else:
        pitch, tempo = 60, 120
        flat_right = ZEROS
        right_present = 0

    # ── Tay trái → chord_type ──
    if 'Left' in hand_dict:
        lms  = hand_dict['Left']
        wy   = lms[0][1]
        row  = min(int(wy * len(CHORD_MAP)), len(CHORD_MAP) - 1)
        chord = CHORD_MAP[row]
        flat_left = [v for lm in normalize_landmarks(lms) for v in lm]
        left_present = 1
    else:
        chord = 0
        flat_left = ZEROS
        left_present = 0

    velocity = 64  # sẽ cập nhật theo tốc độ ở vòng lặp chính

    return pitch, velocity, tempo, chord, right_present, left_present, flat_right, flat_left

# ── Vẽ mũi tên hướng dẫn ─────────────────────────────────────────────────
def draw_arrow(frame, arrow_type, color):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    alpha = int(160 + 95 * abs(np.sin(time.time() * 3)))
    ov = frame.copy()

    lc = (80, 200, 255)   # màu tay trái
    rc = (255, 160, 0)    # màu tay phải

    if arrow_type == "RIGHT_UD":
        cv2.arrowedLine(ov, (cx + 120, cy + 80), (cx + 120, cy - 80), rc, 6, tipLength=0.3)
        cv2.arrowedLine(ov, (cx + 120, cy - 80), (cx + 120, cy + 80), rc, 6, tipLength=0.3)
        cv2.putText(ov, "PHAI", (cx + 85, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc, 2)
        cv2.putText(ov, "DUNG YEN", (cx - 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1)

    elif arrow_type == "RIGHT_FAST":
        for i, ox in enumerate([-50, 0, 50]):
            cv2.arrowedLine(ov, (cx + 60 + ox, cy), (cx + 160 + ox, cy), rc, 5, tipLength=0.3)
        cv2.putText(ov, "VAY NHANH →", (cx + 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc, 2)
        cv2.putText(ov, "DUNG YEN", (cx - 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1)

    elif arrow_type == "LEFT_UD":
        cv2.arrowedLine(ov, (cx - 120, cy + 80), (cx - 120, cy - 80), lc, 6, tipLength=0.3)
        cv2.arrowedLine(ov, (cx - 120, cy - 80), (cx - 120, cy + 80), lc, 6, tipLength=0.3)
        cv2.putText(ov, "TRAI", (cx - 155, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)
        cv2.putText(ov, "DUNG YEN", (cx + 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 1)

    elif arrow_type == "LEFT_CHORD":
        labels = [("MAJOR", cy - 80), ("DOMINANT", cy), ("MINOR", cy + 80)]
        for txt, y in labels:
            cv2.putText(ov, txt, (cx - 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lc, 2)
        cv2.arrowedLine(ov, (cx - 120, cy - 60), (cx - 120, cy + 60), lc, 5, tipLength=0.2)
        cv2.putText(ov, "DUNG YEN", (cx + 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 1)

    elif arrow_type == "BOTH_SAME":
        cv2.arrowedLine(ov, (cx - 100, cy + 70), (cx - 100, cy - 70), color, 5, tipLength=0.3)
        cv2.arrowedLine(ov, (cx + 100, cy + 70), (cx + 100, cy - 70), color, 5, tipLength=0.3)
        cv2.putText(ov, "TRAI", (cx - 135, cy + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)
        cv2.putText(ov, "PHAI", (cx + 75, cy + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 2)

    elif arrow_type == "BOTH_OPP":
        cv2.arrowedLine(ov, (cx - 100, cy + 70), (cx - 100, cy - 70), lc, 5, tipLength=0.3)
        cv2.arrowedLine(ov, (cx + 100, cy - 70), (cx + 100, cy + 70), rc, 5, tipLength=0.3)
        cv2.putText(ov, "TRAI↑", (cx - 135, cy + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)
        cv2.putText(ov, "PHAI↓", (cx + 75, cy + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 2)

    elif arrow_type == "NONE":
        cv2.putText(ov, "HA TAY XUONG", (cx - 140, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

    # ── Arrow types moi ───────────────────────────────────────────────────
    elif arrow_type == "RIGHT_CIRCLE":
        # Ve vong tron ben phai
        cv2.circle(ov, (cx + 120, cy), 60, rc, 3)
        cv2.arrowedLine(ov, (cx + 120, cy - 60), (cx + 180, cy - 60), rc, 4, tipLength=0.4)
        cv2.putText(ov, "VE VONG TRON", (cx + 50, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
        cv2.putText(ov, "DUNG YEN", (cx - 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1)

    elif arrow_type == "RIGHT_ZD":
        # Mui ten ra vao (truc Z - dai dien boi kich thuoc tay)
        cv2.arrowedLine(ov, (cx + 60, cy), (cx + 180, cy), rc, 5, tipLength=0.25)
        cv2.arrowedLine(ov, (cx + 180, cy + 30), (cx + 60, cy + 30), rc, 5, tipLength=0.25)
        cv2.putText(ov, "DAY RA", (cx + 60, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
        cv2.putText(ov, "KEO VAO", (cx + 60, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
        cv2.putText(ov, "DUNG YEN", (cx - 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1)

    elif arrow_type == "LEFT_ZONES":
        # 3 vung cao/giua/thap
        zone_h = h // 3
        for i, (label, c) in enumerate([("TREN - MAJOR",(0,200,100)),
                                         ("GIUA - DOM",  (0,180,255)),
                                         ("DUOI - MINOR",(200,80,255))]):
            y_zone = zone_h * i + zone_h // 2
            cv2.rectangle(ov, (cx-180, zone_h*i+5), (cx-10, zone_h*(i+1)-5), c, 2)
            cv2.putText(ov, label, (cx-175, y_zone+8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.putText(ov, "GIU 2 GIAY MOI VUNG", (cx-180, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        cv2.putText(ov, "DUNG YEN", (cx + 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 1)

    elif arrow_type == "BOTH_EXPAND":
        # 2 tay mo ra thu vao
        cv2.arrowedLine(ov, (cx - 40, cy), (cx - 150, cy), lc, 5, tipLength=0.25)
        cv2.arrowedLine(ov, (cx + 40, cy), (cx + 150, cy), rc, 5, tipLength=0.25)
        cv2.arrowedLine(ov, (cx - 150, cy+35), (cx - 40, cy+35), lc, 4, tipLength=0.25)
        cv2.arrowedLine(ov, (cx + 150, cy+35), (cx + 40, cy+35), rc, 4, tipLength=0.25)
        cv2.putText(ov, "MO RA", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(ov, "THU VAO", (cx-30, cy+70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    elif arrow_type == "RIGHT_FAST_LEFT_HOLD":
        # Tay trai giu giua, tay phai vay
        cv2.circle(ov, (cx - 120, cy), 30, lc, 3)
        cv2.putText(ov, "GIU YEN", (cx-165, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)
        for ox2 in [-50, 0, 50]:
            cv2.arrowedLine(ov, (cx+60+ox2, cy), (cx+130+ox2, cy), rc, 4, tipLength=0.3)
        cv2.putText(ov, "VAY NHANH", (cx+50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)

    elif arrow_type == "RIGHT_ZZ":
        # Zigzag
        pts = [(cx+60, cy+60),(cx+80, cy-60),(cx+100, cy+60),(cx+120, cy-60),(cx+140, cy+60)]
        for i in range(len(pts)-1):
            cv2.arrowedLine(ov, pts[i], pts[i+1], rc, 4, tipLength=0.3)
        cv2.putText(ov, "ZIGZAG NHANH", (cx+40, cy+90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
        cv2.putText(ov, "DUNG YEN", (cx-200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1)

    cv2.addWeighted(ov, alpha / 255.0, frame, 1 - alpha / 255.0, 0, frame)

def draw_chord_zones(frame, chord):
    """Vẽ vùng chord bên phải màn hình."""
    h, w = frame.shape[:2]
    labels = [("MAJOR", (0, 200, 100)), ("MAJOR", (0, 200, 100)),
              ("DOM",   (0, 180, 255)), ("MINOR", (200, 80, 255)),
              ("MINOR", (200, 80, 255))]
    chord_names = {0: "", 1: "MAJOR", 2: "MINOR", 3: "DOM"}
    for i, (name, c) in enumerate(labels):
        y = int(h * i / len(labels))
        cv2.line(frame, (w - 55, y), (w, y), c, 1)
        cv2.putText(frame, name, (w - 52, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
    # hiện chord hiện tại
    cv2.putText(frame, chord_names.get(chord, ""),
                (w - 70, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_pitch_zones(frame):
    h = frame.shape[0]
    note_names = ["C5", "A4", "F4", "D4", "C4"]
    for i, name in enumerate(note_names):
        y = int(h * i / len(note_names))
        cv2.line(frame, (0, y), (45, y), (150, 150, 255), 1)
        cv2.putText(frame, name, (2, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 255), 1)

def draw_hud(frame, mode, mode_idx, sample_count, buf_len,
             pitch, velocity, tempo, chord, rp, lp):
    h, w = frame.shape[:2]
    color = mode["color"]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    # Header
    cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.putText(frame, f"[{mode_idx+1}/{len(MODES)}] {mode['name']}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, mode["desc"],
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Trạng thái tay
    rcolor = (0, 220, 80) if rp else (60, 60, 60)
    lcolor = (0, 180, 255) if lp else (60, 60, 60)
    cv2.putText(frame, f"TAY PHAI: {'ON' if rp else 'OFF'}",
                (w - 210, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rcolor, 2)
    cv2.putText(frame, f"TAY TRAI: {'ON' if lp else 'OFF'}",
                (w - 210, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcolor, 2)

    # Thông số nhạc
    cv2.putText(frame, f"P:{pitch} V:{velocity} T:{tempo} C:{chord}",
                (w - 210, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    draw_pitch_zones(frame)
    draw_chord_zones(frame, chord)

    # Duong ngang vung ly tuong (20% - 80% chieu cao)
    y_top = int(h * 0.20)
    y_bot = int(h * 0.80)
    cv2.line(frame, (55, y_top), (w - 60, y_top), (0, 255, 200), 1)
    cv2.line(frame, (55, y_bot), (w - 60, y_bot), (0, 255, 200), 1)
    cv2.putText(frame, "VUNG LY TUONG", (w // 2 - 70, y_top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
    cv2.putText(frame, "GIU TAY TRONG VUNG NAY", (w // 2 - 100, y_bot + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

    # Buffer bar
    bw = int((w - 10) * buf_len / FRAMES_PER_SAMPLE)
    cv2.rectangle(frame, (5, h - 45), (w - 5, h - 28), (40, 40, 40), -1)
    cv2.rectangle(frame, (5, h - 45), (5 + bw, h - 28), (0, 140, 255), -1)
    cv2.putText(frame, f"Buffer {buf_len}/{FRAMES_PER_SAMPLE}",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

    # Progress bar
    pw = int((w - 10) * sample_count / SAMPLES_PER_MODE)
    cv2.rectangle(frame, (5, h - 25), (w - 5, h - 5), (40, 40, 40), -1)
    cv2.rectangle(frame, (5, h - 25), (5 + pw, h - 5), (0, 200, 80), -1)
    cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_PER_MODE}  |  SPACE=tam dung  Q=thoat",
                (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def show_intro(cap, mode, mode_idx):
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ov = np.zeros_like(frame)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        color = mode["color"]
        cv2.putText(frame, f"CHE DO {mode_idx+1}/{len(MODES)}",
                    (w//2 - 110, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, mode["name"],
                    (w//2 - 200, h//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, mode["desc"],
                    (w//2 - 220, h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
        cv2.putText(frame, "Nhan SPACE de bat dau",
                    (w//2 - 160, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)
        cv2.imshow("GestuRhythm - Thu Data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Dem nguoc 3-2-1
            for count in [3, 2, 1]:
                ret2, frame2 = cap.read()
                if ret2:
                    frame2 = cv2.flip(frame2, 1)
                    h2, w2 = frame2.shape[:2]
                    cv2.putText(frame2, str(count), (w2//2 - 40, h2//2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, mode["color"], 8)
                    cv2.imshow("GestuRhythm - Thu Data", frame2)
                    cv2.waitKey(1000)
            return True
        if key == ord('q'):
            return False

def collect_mode(cap, writer, mode, mode_idx, person_id="unknown"):
    if not show_intro(cap, mode, mode_idx):
        return -1

    sample_count = 0
    frame_buffer = []
    prev_right, prev_time = None, None
    paused = False
    pitch, velocity, tempo, chord = 60, 64, 120, 0
    rp, lp = 0, 0

    while sample_count < SAMPLES_PER_MODE:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        if not paused:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_dict = get_hand_dict(results)

            # Vẽ landmarks lên frame
            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            pitch, velocity, tempo, chord, rp, lp, flat_r, flat_l = auto_label(hand_dict)

            # Tính velocity từ tốc độ tay phải
            if 'Right' in hand_dict:
                wx = hand_dict['Right'][0][0]
                wy = hand_dict['Right'][0][1]
                dt = (now - prev_time) if prev_time else 0.033
                if prev_right:
                    speed = np.hypot(wx - prev_right[0], wy - prev_right[1]) / dt
                    velocity = int(np.clip(40 + speed * 300, 40, 127))
                prev_right = (wx, wy)
                prev_time = now
            else:
                prev_right = None

            frame_buffer.append([now, pitch, velocity, tempo, chord, rp, lp, person_id] + flat_r + flat_l)

            if len(frame_buffer) >= FRAMES_PER_SAMPLE:
                # Kiểm tra tay active có đủ không
                active = mode["active"]
                if active == "right":
                    valid = sum(r[5] for r in frame_buffer) >= 18
                elif active == "left":
                    valid = sum(r[6] for r in frame_buffer) >= 15
                elif active == "both":
                    valid = (sum(r[5] for r in frame_buffer) >= 10 and
                             sum(r[6] for r in frame_buffer) >= 10)
                else:  # none
                    valid = sum(r[5] + r[6] for r in frame_buffer) <= 5

                if valid:
                    for row in frame_buffer:
                        writer.writerow(row)
                    sample_count += 1
                frame_buffer.clear()

        draw_arrow(frame, mode["arrow"], mode["color"])
        draw_hud(frame, mode, mode_idx, sample_count, len(frame_buffer),
                 pitch, velocity, tempo, chord, rp, lp)

        if paused:
            h2, w2 = frame.shape[:2]
            cv2.putText(frame, "TAM DUNG - SPACE de tiep tuc",
                        (w2//2 - 200, h2//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("GestuRhythm - Thu Data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return -1
        elif key == ord(' '):
            paused = not paused

    return sample_count

def delete_mode_data(selected_indices):
    """Xoa data cua cac che do duoc chon trong CSV."""
    if not os.path.exists(OUTPUT_CSV):
        return
    import io
    with open(OUTPUT_CSV, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Xac dinh rows thuoc che do nao theo active hand
    def row_matches_mode(row, mode):
        try:
            rp = int(row[5])
            lp = int(row[6])
            active = mode["active"]
            if active == "right":
                return rp == 1 and lp == 0
            elif active == "left":
                return rp == 0 and lp == 1
            elif active == "both":
                return rp == 1 and lp == 1
            else:  # none
                return rp == 0 and lp == 0
        except Exception:
            return False

    modes_to_delete = [MODES[i] for i in selected_indices]
    kept = [r for r in rows if not any(row_matches_mode(r, m) for m in modes_to_delete)]

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)

    deleted = len(rows) - len(kept)
    print(f"Da xoa {deleted} dong data cu cua che do da chon.")

def main():
    cam_idx = load_camera_index()
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"Không mở được camera {cam_idx}!")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)

    # Header: timestamp + labels + 63 tay phải + 63 tay trái = 133 cột
    lm_r = [f"r_lm{i}_{ax}" for i in range(21) for ax in ('x', 'y', 'z')]
    lm_l = [f"l_lm{i}_{ax}" for i in range(21) for ax in ('x', 'y', 'z')]
    headers = ['timestamp', 'pitch', 'velocity', 'tempo', 'chord_type',
               'right_present', 'left_present', 'person_id'] + lm_r + lm_l

    # Hoi ten nguoi thu
    person_id = input("Nhap ten/ma nguoi thu (vd: nguoi1, A, B): ").strip()
    if not person_id:
        person_id = "unknown"
    print(f"\nNguoi thu: {person_id}")

    # Menu chon che do
    print("\n=== GestuRhythm - Thu Data ===\n")
    print("Cac che do:")
    for i, m in enumerate(MODES):
        print(f"  [{i+1}] {m['name']}")
    print(f"  [0] Thu tat ca\n")

    choice = input("Chon (0=tat ca, hoac nhap so nhu 1,3,5): ").strip()
    if choice == '0' or choice == '':
        selected = list(range(len(MODES)))
    else:
        try:
            idxs = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [i for i in idxs if 0 <= i < len(MODES)]
            if not selected:
                selected = list(range(len(MODES)))
        except ValueError:
            selected = list(range(len(MODES)))

    print(f"\nSe thu lai: {', '.join(MODES[i]['name'] for i in selected)}")
    if os.path.exists(OUTPUT_CSV):
        print("Xoa data cu cua cac che do nay truoc khi thu lai...")
        delete_mode_data(selected)
    print()

    total = 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)

        for i in selected:
            mode = MODES[i]
            print(f"[{i+1}/{len(MODES)}] {mode['name']}")
            count = collect_mode(cap, writer, mode, i, person_id)
            if count == -1:
                print("Da thoat.")
                break
            total += count
            print(f"  {count} samples\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tong: {total} samples -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
