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

> **Luu y:** Can Python 3.11 (khong dung 3.12+). Kiem tra bang: `py -3.11 --version`
> Neu chua co: tai tai https://www.python.org/downloads/release/python-3119/

### Cai FluidSynth (can cho phan am thanh)
1. Tai file zip tai: https://github.com/FluidSynth/fluidsynth/releases
   - Chon file: `fluidsynth-2.x.x-win10-x64-glib.zip`
2. Giai nen, copy tat ca file trong thu muc `bin\` vao `C:\tools\fluidsynth\bin\`
3. Tai SoundFont: tim "FluidR3_GM.sf2" tren Google (file ~140MB)
   - Luu vao: `soundfonts\FluidR3_GM.sf2`

### Cai Camera (neu webcam hong)
- **Camo** (iPhone): tai Camo Studio cho Windows + app Camo tren iPhone
- **DroidCam** (Android/iPhone): tai tai dev47apps.com, cai ca app va client PC

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
# 1. Merge data tu nhieu nguoi
python scripts/merge_data.py   # (neu co nhieu file CSV)

# 2. Prepare dataset
python scripts/prepare_dataset_v2.py
# -> data/processed/gesturhythm_v2.zip

# 3. Upload gesturhythm_v2.zip len Kaggle Dataset (ten: gesturhythm-v2)

# 4. Train theo thu tu:
#    Buoc 1: notebooks/train_v2.ipynb          -> gesture_emotion_encoder.pt
#    Buoc 2: notebooks/train_music_prior.ipynb -> music_prior.pt
#    Buoc 3: notebooks/train_conditioned_decoder.ipynb -> conditioned_decoder.pt
#    (Buoc 3 can ca 2 model tren + gesturhythm-v2 dataset)

# 5. Download ve, luu vao model/
#    model/gesture_emotion_encoder.pt
#    model/music_prior.pt
#    model/conditioned_decoder.pt

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
