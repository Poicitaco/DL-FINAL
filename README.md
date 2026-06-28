# GestuRhythm

He thong tong hop am nhac real-time bang cu chi tay - Su dung Transformer va MIDI.

## Setup

### 1. Yeu cau
- Python 3.11
- Webcam (hoac Camo/DroidCam qua dien thoai)

### 2. Cai dat
```bash
git clone https://github.com/Poicitaco/DL-FINAL.git
cd DL-FINAL

py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Cai FluidSynth (cho inference)
- Tai: https://github.com/FluidSynth/fluidsynth/releases
- Giai nen vao C:\tools\fluidsynth\bin\
- Tai SoundFont: FluidR3_GM.sf2 -> luu vao soundfonts/

### 4. Thu data
```bash
venv\Scripts\activate
python scripts/select_camera.py   # chon camera
python scripts/collect_data.py    # thu data (nhap ten ban)
```

### 5. Chuan bi dataset
```bash
python scripts/prepare_dataset.py
# Upload data/processed/ len Kaggle
```

### 6. Train tren Kaggle
- Upload model/train_seq2seq.ipynb len Kaggle Notebook
- Them dataset gesturhythm
- Chay voi GPU T4x2
- Download gesture_seq2seq.pt ve model/

### 7. Chay real-time
```bash
python inference/realtime.py
```

## Cau truc project
```
scripts/
  select_camera.py   - Chon camera
  collect_data.py    - Thu data landmarks (11 che do)
  verify_live.py     - Kiem tra live landmarks
  prepare_dataset.py - Xu ly data cho Kaggle

model/
  train_seq2seq.ipynb  - Notebook train Seq2Seq Transformer

inference/
  realtime.py          - Real-time inference + MIDI output

data/
  raw/                 - gesture_data.csv (sau khi thu)
  processed/           - sequences.npy, labels.npy (sau prepare)

soundfonts/            - FluidR3_GM.sf2 (tu tai ve)
```

## Ghi chu khi thu data
- Dung tay phai lam melody, tay trai lam chord
- Mo tay ra, khong nam tay
- Cam camera cach 50-80cm
- Moi nguoi thu het 11 che do x 150 samples
