# GestuRhythm v2

Hệ thống tổng hợp âm nhạc real-time dựa trên cảm xúc của cử chỉ tay.

Mô hình AI học tách biệt hai phần độc lập:
1. **Emotion Encoder**: Nhận diện cảm xúc từ cử chỉ tay.
2. **Music Prior**: Học ngữ pháp âm nhạc từ dữ liệu MIDI thực tế.

Sau đó, hệ thống kết hợp cả hai thành phần trên để sinh ra âm nhạc vừa truyền tải được cảm xúc, vừa đảm bảo đúng nhạc lý.

---

## Kiến trúc hệ thống (v2)

```text
Webcam
  -> MediaPipe Holistic (21 hand + 33 pose landmarks)
  -> Gesture Emotion Encoder (Transformer) -> Vector cảm xúc [valence, arousal]
  -> Conditioned Music Decoder (Cross-Attention)
  -> Chuỗi 16 nốt nhạc
  -> Scale Mask (đảm bảo đúng nhạc lý)
  -> FluidSynth -> Đầu ra âm thanh
         ^
         |
  Music Prior (Mô hình GPT huấn luyện trên tập dữ liệu Lakh MIDI)
```

---

## Hướng dẫn cài đặt (Chỉ cần thực hiện một lần)

### 1. Khởi tạo môi trường

Chạy các lệnh sau trong terminal để tải mã nguồn và cài đặt các thư viện cần thiết:

```bash
git clone https://github.com/Poicitaco/DL-FINAL.git
cd DL-FINAL

# Tạo và kích hoạt môi trường ảo Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> [!IMPORTANT]
> Dự án yêu cầu phiên bản **Python 3.11** (không hỗ trợ Python 3.12+).
> - Kiểm tra phiên bản hiện tại bằng lệnh: `py -3.11 --version`
> - Nếu chưa cài đặt, vui lòng tải về tại: [Python 3.11.9 Downloads](https://www.python.org/downloads/release/python-3119/)

### 2. Cài đặt FluidSynth (bắt buộc cho đầu ra âm thanh)

1. Tải bản phân phối FluidSynth dưới dạng file zip từ [FluidSynth Releases](https://github.com/FluidSynth/fluidsynth/releases).
   - Chọn tệp: `fluidsynth-2.x.x-win10-x64-glib.zip` (phù hợp với hệ điều hành Windows 10/11 x64).
2. Giải nén và sao chép toàn bộ các tệp trong thư mục `bin\` vào đường dẫn hệ thống: `C:\tools\fluidsynth\bin\`.
3. Tải tệp SoundFont: Tìm kiếm từ khóa "FluidR3_GM.sf2" trên Google (dung lượng tệp khoảng 140MB).
   - **Cụ thể, bạn có thể tải tại đây:** [FluidR3_GM.sf2 (136 MB)](https://sourceforge.net/projects/pianobooster/)
   - Lưu tệp đã tải vào thư mục dự án theo đường dẫn: `soundfonts/FluidR3_GM.sf2`.

### 3. Cài đặt Camera thay thế (nếu không có webcam tích hợp)

Nếu webcam của máy tính gặp sự cố hoặc không khả dụng, bạn có thể biến điện thoại thành camera phụ thông qua các ứng dụng sau:
- **Camo (iOS):** Tải phần mềm *Camo Studio* cho Windows và cài đặt ứng dụng *Camo* tương ứng trên App Store của iPhone.
- **DroidCam (Android/iOS):** Tải phần mềm client từ trang [dev47apps.com](https://www.dev47apps.com/) cho máy tính và cài đặt ứng dụng trên thiết bị di động.

---

## Thu thập dữ liệu cử chỉ (Tất cả thành viên thực hiện)

Chạy các kịch bản sau để thiết lập camera và tiến hành thu thập dữ liệu cử chỉ tay:

```bash
# Kích hoạt môi trường ảo nếu chưa thực hiện
venv\Scripts\activate

# Chọn nguồn camera đầu vào (chỉ cần làm một lần)
python scripts/select_camera.py

# Bắt đầu chạy script thu thập dữ liệu cử chỉ
python scripts/collect_data_v2.py
```

### 6 trạng thái cảm xúc cần thu thập

| # | Tên nhãn | Cảm xúc tương ứng | Cách di chuyển tay |
|---|---|---|---|
| 1 | `HAPPY_HIGH` | Vui vẻ + Năng lượng cao | Di chuyển tay **nhanh, biên độ rộng, đưa lên cao** (tương tự như đang nhảy múa). |
| 2 | `HAPPY_LOW` | Vui vẻ + Nhẹ nhàng | Di chuyển tay **chậm, vẽ đường tròn nhẹ nhàng** (tương tự như đang lau kính). |
| 3 | `SAD_HIGH` | Buồn bã + Căng thẳng | Đặt tay **thấp**, di chuyển **gấp gáp và co rút** lại (như đang bóp tay). |
| 4 | `SAD_LOW` | Buồn bã + Trầm lắng | Đặt tay **thấp nhất**, di chuyển **rất chậm và nặng nề**. |
| 5 | `NEUTRAL` | Trung tính | Giữ tay ở **giữa màn hình**, di chuyển **đều đặn** (tương tự như đang gõ bàn phím). |
| 6 | `NONE` | Không có cử chỉ | Hạ tay xuống hoàn toàn, **không để tay** xuất hiện trong khung hình. |

### Lưu ý quan trọng khi thu thập dữ liệu cử chỉ

- **Sử dụng một tay:** Chỉ cần sử dụng một tay thuận (không cần thực hiện bằng cả hai tay).
- **Vị trí cơ thể:** Đảm bảo nửa thân trên nằm trong khung hình (MediaPipe tính toán các điểm mốc bắt đầu từ khớp vai trở xuống).
- **Tư thế tay:** Mở rộng bàn tay (không nắm lại), hướng lòng bàn tay về phía camera.
- **Khoảng cách:** Đặt camera cách người từ 50-70 cm, đảm bảo nhìn thấy rõ từ vai đến các ngón tay.
- **Vùng giới hạn:** Giữ bàn tay nằm trong phạm vi giữa hai đường kẻ xanh hiển thị trên màn hình.
- **Tương tác cảm xúc:** **Hãy lắng nghe nhạc nền và di chuyển tay hòa nhịp theo cảm xúc của bản nhạc**.
- **Tiến trình:** Mỗi chế độ thu thập dữ liệu sẽ tự động dừng sau khi lưu đủ 150 mẫu (samples). Nhấn phím `SPACE` để chuyển sang chế độ tiếp theo.
- **Dừng chương trình:** Nhấn phím `Q` để thoát (dữ liệu đã thu thập trước đó vẫn sẽ được lưu trữ tự động).

### Hướng dẫn chi tiết từng chế độ cử chỉ

| # | Tên nhãn | Hành động cụ thể |
|---|---|---|
| 1 | `HAPPY_HIGH` | Giữ tay lên cao, vẫy qua lại **nhanh** và **rộng** (tương tự như đang cổ vũ). |
| 2 | `HAPPY_LOW` | Di chuyển tay theo chiều ngang một cách **chậm rãi**, biên độ nhỏ (tương tự như đang vuốt ve). |
| 3 | `SAD_HIGH` | Đặt tay **thấp**, di chuyển **gấp gáp** và co rút tay lại (tương tự như đang bóp tay). |
| 4 | `SAD_LOW` | Đặt tay **thấp nhất**, di chuyển **rất chậm** (cảm giác tay nặng nề không nhấc lên nổi). |
| 5 | `NEUTRAL` | Giữ tay ở **giữa màn hình**, di chuyển **đều đặn** (tương tự như đang gõ bàn phím). |
| 6 | `NONE` | Hạ tay xuống, **không để tay** xuất hiện trong khung hình camera. |

### Lưu trữ và chia sẻ dữ liệu

Sau khi hoàn tất quá trình thu thập, hãy tải tệp dữ liệu `data/raw/gesture_data_v2.csv` lên thư mục Google Drive chung của nhóm.
- **Quy ước đặt tên tệp:** Đặt tên tệp theo tên thành viên để tránh trùng lặp: `gesture_data_v2_A.csv`, `gesture_data_v2_B.csv`,...

---

## Quy trình huấn luyện và chạy thử nghiệm (Dành cho trưởng nhóm/người tích hợp)

Khi đã thu thập đầy đủ tệp tin CSV từ các thành viên, thực hiện các bước sau để gộp dữ liệu, tiền xử lý, huấn luyện và chạy thử nghiệm:

```bash
# 1. Di chuyển tất cả các tệp CSV nhận được vào thư mục: data/raw/

# 2. Gộp dữ liệu từ các thành viên
python scripts/merge_data.py

# 3. Tiền xử lý dữ liệu để chuẩn bị cho việc huấn luyện
python scripts/prepare_dataset_v2.py

# 4. Tải dữ liệu đã xử lý lên Kaggle và thực hiện huấn luyện mô hình theo notebook:
# notebooks/train_v2.ipynb

# 5. Sau khi huấn luyện thành công, tải các file trọng số mô hình đã train về thư mục model/:
# - gesture_emotion_encoder.pt
# - music_prior.pt
# - conditioned_decoder.pt

# 6. Chạy thử nghiệm hệ thống trong thời gian thực (real-time)
python inference/realtime_v2.py
```

---

## Cấu trúc thư mục dự án

```text
.
├── src/
│   ├── emotion_encoder/     # Kiến trúc mô hình Gesture Emotion Encoder
│   ├── music_prior/         # Kiến trúc mô hình GPT Music Prior
│   ├── decoder/             # Kiến trúc mô hình Conditioned Music Decoder
│   └── utils/               # Các hàm bổ trợ (Scale mask, xử lý MIDI...)
│
├── scripts/
│   ├── select_camera.py     # Script cấu hình và chọn thiết bị camera đầu vào
│   ├── collect_data_v2.py   # Script thu thập dữ liệu cử chỉ (6 chế độ cảm xúc)
│   ├── merge_data.py        # Script gộp dữ liệu từ nhiều thành viên
│   └── prepare_dataset_v2.py # Script tiền xử lý dữ liệu chuẩn bị cho việc train trên Kaggle
│
├── data/
│   ├── raw/                 # Chứa tệp gesture_data_v2.csv sau khi thu thập xong
│   └── processed/           # Chứa tập dữ liệu train/val đã được phân tách
│
├── notebooks/
│   └── train_v2.ipynb       # Jupyter notebook dùng để huấn luyện mô hình trên Kaggle
│
├── model/                   # Thư mục lưu trữ các trọng số mô hình đã huấn luyện
│   ├── gesture_emotion_encoder.pt
│   └── conditioned_decoder.pt
│
├── inference/
│   └── realtime_v2.py       # Script chạy thực nghiệm hệ thống thời gian thực (demo real-time)
│
└── soundfonts/
    └── FluidR3_GM.sf2       # Tệp cơ sở dữ liệu âm thanh SoundFont (tải thủ công)
```
