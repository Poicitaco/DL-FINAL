"""
Hien thi truc tiep landmarks + label de kiem tra data thu dung khong.
Khong luu gi ca - chi de xem.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time

CAMERA_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'camera_config.txt')
PITCH_MAP = [72, 69, 65, 62, 60]
CHORD_MAP = [1, 1, 3, 2, 2]
CHORD_NAMES = {0: "NONE", 1: "MAJOR", 2: "MINOR", 3: "DOM"}

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5)

def load_cam():
    try:
        return int(open(CAMERA_CONFIG).read().strip())
    except Exception:
        return 0

def get_hand_dict(results):
    d = {}
    if not results.multi_hand_landmarks:
        return d
    for lms, info in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = info.classification[0].label
        d[label] = [(lm.x, lm.y, lm.z) for lm in lms.landmark]
    return d

cap = cv2.VideoCapture(load_cam(), cv2.CAP_MSMF)
prev_right, prev_time = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hand_dict = get_hand_dict(results)

    # Ve landmarks
    if results.multi_hand_landmarks:
        for lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

    # Tinh label
    pitch, velocity, tempo, chord = 0, 0, 0, 0
    rp, lp = 0, 0

    if 'Right' in hand_dict:
        lms = hand_dict['Right']
        wx, wy = lms[0][0], lms[0][1]
        finger_y = lms[12][1]
        row   = min(int(finger_y * len(PITCH_MAP)), len(PITCH_MAP) - 1)
        pitch = PITCH_MAP[row]
        tempo = int(np.clip(80 + wx * 80, 80, 160))
        dt = (now - prev_time) if prev_time else 0.033
        if prev_right:
            speed = np.hypot(wx - prev_right[0], wy - prev_right[1]) / dt
            velocity = int(np.clip(40 + speed * 300, 40, 127))
        else:
            velocity = 64
        prev_right, prev_time = (wx, wy), now
        rp = 1
    else:
        prev_right = None

    if 'Left' in hand_dict:
        lms = hand_dict['Left']
        wy  = lms[0][1]
        row = min(int(wy * len(CHORD_MAP)), len(CHORD_MAP) - 1)
        chord = CHORD_MAP[row]
        lp = 1

    # Ve vung pitch ben trai
    note_names = ["C5", "A4", "F4", "D4", "C4"]
    for i, name in enumerate(note_names):
        y = int(h * i / len(note_names))
        cv2.line(frame, (0, y), (45, y), (150, 150, 255), 1)
        cv2.putText(frame, name, (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 255), 1)

    # Ve vung chord ben phai
    chord_labels = [("MAJOR",(0,200,100)),("MAJOR",(0,200,100)),
                    ("DOM",(0,180,255)),("MINOR",(200,80,255)),("MINOR",(200,80,255))]
    for i, (name, c) in enumerate(chord_labels):
        y = int(h * i / len(chord_labels))
        cv2.line(frame, (w-55, y), (w, y), c, 1)
        cv2.putText(frame, name, (w-52, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1)

    # Ve mui ten huong dan
    cx, cy = w // 2, h // 2
    alpha = int(160 + 95 * abs(np.sin(now * 3)))
    ov = frame.copy()

    # Tay phai: len xuong (ben phai man hinh)
    if rp:  # dang co tay -> hien mui ten mo nhat
        arrow_color_r = (80, 100, 150)
    else:
        arrow_color_r = (255, 160, 0)  # cam sang khi chua co tay
    cv2.arrowedLine(ov, (cx + 150, cy + 80), (cx + 150, cy - 80), arrow_color_r, 5, tipLength=0.3)
    cv2.arrowedLine(ov, (cx + 150, cy - 80), (cx + 150, cy + 80), arrow_color_r, 5, tipLength=0.3)
    cv2.putText(ov, "TAY PHAI", (cx + 110, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color_r, 2)

    # Tay trai: len xuong (ben trai man hinh)
    if lp:
        arrow_color_l = (80, 100, 150)
    else:
        arrow_color_l = (80, 200, 255)  # xanh sang khi chua co tay
    cv2.arrowedLine(ov, (cx - 150, cy + 80), (cx - 150, cy - 80), arrow_color_l, 5, tipLength=0.3)
    cv2.arrowedLine(ov, (cx - 150, cy - 80), (cx - 150, cy + 80), arrow_color_l, 5, tipLength=0.3)
    cv2.putText(ov, "TAY TRAI", (cx - 195, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color_l, 2)

    cv2.addWeighted(ov, alpha / 255.0, frame, 1 - alpha / 255.0, 0, frame)

    # Duong ngang vung ly tuong
    y_top = int(h * 0.20)
    y_bot = int(h * 0.80)
    cv2.line(frame, (55, y_top), (w - 60, y_top), (0, 255, 200), 1)
    cv2.line(frame, (55, y_bot), (w - 60, y_bot), (0, 255, 200), 1)
    cv2.putText(frame, "VUNG LY TUONG - GIU TAY TRONG DAY",
                (w // 2 - 160, y_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

    # Panel thong so
    cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 0), -1)

    rc = (0, 220, 80) if rp else (80, 80, 80)
    lc = (0, 180, 255) if lp else (80, 80, 80)
    cv2.putText(frame, f"TAY PHAI [{('ON' if rp else 'OFF')}]  Pitch:{pitch}  Vel:{velocity}  Tempo:{tempo}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
    cv2.putText(frame, f"TAY TRAI [{('ON' if lp else 'OFF')}]  Chord:{chord} ({CHORD_NAMES[chord]})",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2)

    # Chi dan
    cv2.putText(frame, "Q = thoat", (w - 100, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Canh bao neu khong thay tay
    if not rp and not lp:
        cv2.putText(frame, "KHONG THAY TAY - Dua tay vao khung hinh",
                    (w//2 - 220, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Verify Live - Kiem tra data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
