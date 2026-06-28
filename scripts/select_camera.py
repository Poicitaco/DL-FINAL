"""
Quet tat ca camera kha dung va cho phep chon.
Ho tro Camo, DroidCam va webcam thuong.
Luu index vao camera_config.txt.
"""
import cv2


def scan_cameras(max_index=8):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, f"Camera {i} [{w}x{h}]"))
            cap.release()
    return available


def main():
    print("=== GestuRhythm - Chon Camera ===\n")
    print("Dang quet camera...")
    cameras = scan_cameras()

    if not cameras:
        print("Khong tim thay camera nao!")
        print("Neu dung Camo/DroidCam: mo app truoc, ket noi USB/WiFi, roi chay lai.")
        return

    print(f"Tim thay {len(cameras)} camera:\n")
    for idx, name in cameras:
        print(f"  [{idx}] {name}")

    print("\nGoi y: Camo/DroidCam thuong la camera co index cao nhat.")

    while True:
        try:
            choice = int(input("\nNhap index camera muon dung: "))
            if any(i == choice for i, _ in cameras):
                break
            print("Index khong hop le.")
        except ValueError:
            print("Nhap so nguyen.")

    # Preview
    cap = cv2.VideoCapture(choice, cv2.CAP_MSMF)
    print(f"\nPreview Camera {choice} — nhan Q de xac nhan chon.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Camera {choice} - Nhan Q de chon",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    confirm = input(f"\nDung Camera {choice}? (y/n): ").strip().lower()
    if confirm == 'y':
        with open("camera_config.txt", "w") as f:
            f.write(str(choice))
        print(f"Da luu Camera {choice} vao camera_config.txt")
    else:
        print("Huy. Chay lai de chon lai.")


if __name__ == "__main__":
    main()
