# GestuRhythm - Mo ta Du an

## Y tuong chinh

He thong cho phep nguoi dung **dieu khien am nhac bang cu chi tay** truoc webcam.
Khong can nhac cu, khong can biet nhac ly — chi can di chuyen tay.

## Demo trong 30 giay

```
Ban giu tay phai truoc camera
  -> He thong phat tien guitar theo vi tri tay
Tay phai di chuyen len cao
  -> Giai dieu cao hon
Tay trai giu o vung tren
  -> Hoa am Major (vui, sang)
Tay trai ha xuong vung duoi
  -> Hoa am Minor (buon, tram)
Tay phai ve vong tron
  -> Vibrato effect
Ha 2 tay xuong
  -> Nhac dung
```

---

## He thong gom 3 phan

### Phan 1 — Thu thap du lieu (lam o may tinh ca nhan)
- Dung webcam quay cu chi tay
- MediaPipe trich xuat 21 diem tren moi ban tay (126 so/frame)
- Luu thanh file CSV

**Viec cua ban:** Chay `collect_data.py`, thu het 11 che do, upload CSV len Drive.

---

### Phan 2 — Huan luyen mo hinh (lam tren Kaggle)
- Mo hinh: **Transformer Encoder-Decoder**
  - Encoder doc 30 frame cu chi -> hieu "ban dang lam gi"
  - Decoder sinh chuoi 16 not nhac -> tao ra "doan nhac tuong ung"
- Giong kien truc GPT nhung sinh not nhac thay vi text

**Viec cua ban:** Khong can lam gi, nhom truong train.

---

### Phan 3 — Chay real-time (demo)
- Webcam -> MediaPipe -> Mo hinh -> Not nhac -> FluidSynth -> Am thanh
- Giao dien web tren Hugging Face Spaces: thay giao vao link la dung duoc

---

## Vai tro moi nguoi

| Viec | Ai lam |
|------|--------|
| Thu data | **Tat ca moi nguoi** (moi nguoi 40 phut) |
| Train model | Nhom truong (tren Kaggle) |
| Viet bao cao | Phan cong sau |
| Demo | Tat ca cung chay |

---

## Cach thu data (chi 3 buoc)

```bash
# 1. Clone code
git clone https://github.com/Poicitaco/DL-FINAL.git
cd DL-FINAL

# 2. Cai dat (chi lam 1 lan)
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Thu data
python scripts/select_camera.py   # chon camera
python scripts/collect_data.py    # thu data, nhap ten ban
```

Sau khi thu xong: upload file `data/raw/gesture_data.csv` len Drive nhe.

---

## Luu y khi thu

- **Mo tay**, khong nam
- **Cach camera 50-80cm**
- **Giu tay trong 2 duong xanh** tren man hinh
- **Nghe nhac** phat ra va di chuyen tay theo cam xuc
- Moi che do **tu dong dung** khi du 150 samples, nhan SPACE de sang che do tiep theo
- Nhan **Q** de thoat bat cu luc nao (data da thu van duoc luu)
