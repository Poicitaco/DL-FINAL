"""
Quét tất cả camera khả dụng và cho phép chọn.
Hỗ trợ Camo (camera ảo từ iPhone).
Lưu index đã chọn vào camera_config.txt để các script khác dùng.
"""
import cv2

def scan_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Lấy tên camera nếu được
                name = cap.getBackendName()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, f"Camera {i} [{w}x{h}] ({name})"))
            cap.release()
    return available

def preview_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    print(f"  Đang preview Camera {index} — nhấn Q để thoát preview")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Camera {index} - Nhan Q de chon", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=== GestuRhythm — Chọn Camera ===\n")
    print("Đang quét camera...")
    cameras = scan_cameras()

    if not cameras:
        print("Không tìm thấy camera nào!")
        print("Nếu dùng Camo: mở app Camo trên iPhone + Mac/PC trước.")
        return

    print(f"Tìm thấy {len(cameras)} camera:\n")
    for idx, name in cameras:
        print(f"  [{idx}] {name}")

    print("\nGợi ý: Camo thường là camera có index cao nhất.")
    print("Nhập index để preview, sau đó nhấn Q để chọn.\n")

    while True:
        try:
            choice = int(input("Nhập index camera muốn dùng: "))
            if any(i == choice for i, _ in cameras):
                break
            print("Index không hợp lệ, thử lại.")
        except ValueError:
            print("Nhập số nguyên.")

    preview_camera(choice)

    confirm = input(f"\nDùng Camera {choice}? (y/n): ").strip().lower()
    if confirm == 'y':
        with open("camera_config.txt", "w") as f:
            f.write(str(choice))
        print(f"Da luu Camera {choice} vao camera_config.txt")
    else:
        print("Hủy. Chạy lại script để chọn lại.")

if __name__ == "__main__":
    main()
