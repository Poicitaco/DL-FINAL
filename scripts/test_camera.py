"""Test tat ca backend va index de tim camera hoat dong."""
import cv2

BACKENDS = [
    ("MSMF",   cv2.CAP_MSMF),
    ("ANY",    cv2.CAP_ANY),
    ("DSHOW",  cv2.CAP_DSHOW),
]

for backend_name, backend in BACKENDS:
    for idx in range(5):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        status = "CO HINH" if (ret and frame is not None) else "DEN / KHONG CO HINH"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{backend_name}] index={idx} {w}x{h} -> {status}")
        if ret and frame is not None:
            cv2.imshow(f"{backend_name} idx={idx}", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        cap.release()

print("\nNhap index va backend muon dung vao camera_config.txt")
