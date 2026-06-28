# Huong dan Thu Data GestuRhythm v2

## Chuan bi

- Webcam (hoac Camo/DroidCam qua dien thoai)
- Dung hoac ngoi cach camera 50-70cm
- Camera huong thang vao mat/than tren — can thay ro tu vai den ban tay
- Phong du anh sang

---

## 6 Che do — Hinh dung de nho

---

### Che do 1: HAPPY_HIGH
**Tuong tuong:** Ban dang o san van dong, doi bong yeu thich vua ghi ban. Ban nhay len co vu.

**Lam gi:** 2 tay giu len CAO, vay qua lai NHANH va RONG. Nhu dang hoan ho, reo ho.

---

### Che do 2: HAPPY_LOW
**Tuong tuong:** Ban dang nghe bai nhac yeu thich, ngoi thu gian, dau hoi lay dong.

**Lam gi:** 2 tay luot nhe ngang, CHAM rai, bieu do VUA. Nhu dang dua tay theo song nhac.

---

### Che do 3: SAD_HIGH
**Tuong tuong:** Ban dang cam hai hop thuoc va XOC XOC — tay thap, chuyen dong nho va giat cuc.

**Lam gi:** 2 tay de THAP (ngang hong), chuyen dong GAP va KHONG DEU. Nhu dang bop bop cai gi do trong tay.

---

### Che do 4: SAD_LOW
**Tuong tuong:** Ban vua khoc xong, met moi, 2 tay THAP LON, nhu khong con suc nac len.

**Lam gi:** 2 tay de THAP NHAT trong khung hinh, di chuyen RAT CHAM. Nhu dang keo le doi tay nang ne.

---

### Che do 5: NEUTRAL
**Tuong tuong:** Ban dang nhin dien thoai, ngoi binh thuong, tay gat gat man hinh.

**Lam gi:** 2 tay o GIUA man hinh, di chuyen DEU — khong qua nhanh, khong qua cham.

---

### Che do 6: NONE
**Lam gi:** Ha ca 2 tay xuong, buong xuoi. Khong co tay trong khung hinh.

---

## Quy trinh moi che do

```
Script hien man hinh gioi thieu
    -> Nhan SPACE
    -> Dem nguoc 3-2-1
    -> Nhac nen bat dau phat
    -> Di chuyen tay THEO CAM XUC trong khi nghe nhac
    -> Thanh tien do day -> tu dong chuyen che do tiep
```

## Luu y quan trong

- **1 tay hoac 2 tay deu duoc** — 2 tay thi data tot hon
- Mo tay, long ban tay huong vao camera
- **Nghe nhac nen va de cam xuc tu nhien** — dung dien xuat qua lo
- Nhan **SPACE** de tam dung neu can nghi
- Nhan **Q** de thoat (data da thu van duoc luu)

## Sau khi thu xong

Upload file `data/raw/gesture_data_v2.csv` len Google Drive.
Doi ten: `gesture_data_v2_[ten].csv` (vd: gesture_data_v2_A.csv)
