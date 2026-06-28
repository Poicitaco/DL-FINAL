# GestuRhythm v2

He thong tong hop am nhac real-time dua tren cam xuc cu chi tay.
AI hoc tach biet: (1) cam xuc tu cu chi va (2) ngu phap am nhac tu MIDI thuc te.
Sau do ket hop de sinh nhac vua co cam xuc vua dung nhac ly.

---

## Kien truc v2

```
Webcam
  -> MediaPipe Holistic (21 hand + 33 pose landmarks)
  -> Gesture Emotion Encoder (Transformer) -> Vector cam xuc [valence, arousal]
  -> Conditioned Music Decoder (Cross-Attention)
  -> 16 not nhac
  -> Scale Mask (dam bao dung nhac ly)
  -> FluidSynth -> Am thanh
         ^
         |
  Music Prior (GPT trained on Lakh MIDI)
```

---

## Cai dat (lam 1 lan)

```bash
git clone https://github.com/Poicitaco/DL-FINAL.git
cd DL-FINAL

py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Cai FluidSynth
- Tai: https://github.com/FluidSynth/fluidsynth/releases
- Giai nen vao C:\tools\fluidsynth\bin\
- Tai SoundFont FluidR3_GM.sf2 -> luu vao soundfonts/

---

## Thu data (moi nguoi lam)

```bash
venv\Scripts\activate
python scripts/select_camera.py    # chon camera 1 lan
python scripts/collect_data_v2.py  # thu data
```

### 6 che do cam xuc can thu

| # | Ten | Cam xuc | Cach di chuyen |
|---|-----|---------|----------------|
| 1 | HAPPY_HIGH | Vui + Nang luong cao | Tay NHANH, RONG, LEN CAO — nhu dang nhay mua |
| 2 | HAPPY_LOW  | Vui + Nhe nhang      | Tay CHAM, TRON, NHE — nhu dang lau kinh |
| 3 | SAD_HIGH   | Buon + Cang thang    | Tay xuong thap, di chuyen GAP va CO RUT lai |
| 4 | SAD_LOW    | Buon + Cham          | Tay xuong thap nhat, RAT CHAM va NANG NE |
| 5 | NEUTRAL    | Trung tinh           | Tay giua man hinh, di chuyen DEU DANG |
| 6 | NONE       | Khong tay            | De tay xuong, KHONG co tay trong khung hinh |

### Luu y khi thu
- **Chi can 1 tay** (tay thuan) — khong can dung ca 2 tay
- **Nua than tren** trong khung hinh — MediaPipe bat dau tu vai tro xuong
- Mo tay (khong nam), long ban tay huong vao camera
- Cam camera cach 50-70cm, thay ro tu vai den ngon tay
- Giu tay trong 2 duong xanh tren man hinh
- **NGHE NHAC NEN va DI CHUYEN TAY THEO CAM XUC CUA NHAC**
- Moi che do tu dung khi du 150 samples, nhan SPACE sang che do tiep
- Nhan Q de thoat (data da thu van duoc luu)

### Huong dan cu the tung che do

| # | Ten | Lam gi cu the |
|---|-----|---------------|
| 1 | HAPPY_HIGH | Tay giu len cao, vay qua lai NHANH va RONG — nhu dang co vu |
| 2 | HAPPY_LOW  | Tay di chuyen ngang CHAM, bieu do nho — nhu dang vuot ve |
| 3 | SAD_HIGH   | Tay de THAP, di chuyen GAP va co rut lai — nhu dang bop tay |
| 4 | SAD_LOW    | Tay de THAP NHAT, di chuyen RAT CHAM — nhu tay nang khong nhac len |
| 5 | NEUTRAL    | Tay o GIUA man hinh, di chuyen DEU DANG — nhu dang go ban phim |
| 6 | NONE       | Ha tay xuong, KHONG co tay trong khung hinh |

### Upload CSV len Drive
Sau khi thu xong, upload file `data/raw/gesture_data_v2.csv` len Google Drive.
Dat ten theo nguoi: `gesture_data_v2_A.csv`, `gesture_data_v2_B.csv`...

---

## Sau khi co du data (nhom truong lam)

```bash
# 1. Dat cac file CSV vao data/raw/
# 2. Merge
python scripts/merge_data.py

# 3. Prepare dataset
python scripts/prepare_dataset_v2.py

# 4. Upload len Kaggle, train notebook
# notebooks/train_v2.ipynb

# 5. Download model ve model/
# gesture_emotion_encoder.pt
# music_prior.pt
# conditioned_decoder.pt

# 6. Chay real-time
python inference/realtime_v2.py
```

---

## Cau truc thu muc

```
src/
  emotion_encoder/   - Gesture Emotion Encoder model
  music_prior/       - GPT Music Prior model
  decoder/           - Conditioned Music Decoder
  utils/             - Scale mask, MIDI utils

scripts/
  select_camera.py          - Chon camera
  collect_data_v2.py        - Thu data (6 che do cam xuc)
  merge_data.py             - Gop data nhieu nguoi
  prepare_dataset_v2.py     - Xu ly data cho Kaggle

data/
  raw/       - gesture_data_v2.csv (sau khi thu)
  processed/ - train/val split (sau khi prepare)

notebooks/
  train_v2.ipynb   - Train tren Kaggle

model/
  gesture_emotion_encoder.pt
  conditioned_decoder.pt

inference/
  realtime_v2.py   - Chay demo real-time

soundfonts/
  FluidR3_GM.sf2   - Am thanh (tu tai ve)
```
