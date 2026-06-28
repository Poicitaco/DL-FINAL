# Huong dan Thu Data GestuRhythm v2

## Chuan bi truoc khi thu

**Dung cu:**
- Webcam (hoac Camo/DroidCam qua dien thoai)
- Phong du anh sang, nen sang sau lung

**Tu the:**
- Ngoi hoac dung cach camera 50-70cm
- **Nua than tren** trong khung hinh — MediaPipe can thay ro tu vai den ngon tay
- Long ban tay huong vao camera
- Chi can **1 tay** (tay thuan)

---

## 6 Che do Can Thu

---

### Che do 1: HAPPY_HIGH (Vui + Nang luong cao)

**Cam xuc:** Nhu dang co vu doi bong yeu thich vua ghi ban.

**Lam gi:**
- Giu tay len cao (vung tren man hinh)
- Vay tay qua lai NHANH — khoang 2-3 lan/giay
- Bien do RONG — tu trai sang phai ~50cm
- Toan than co the hoi lay dong theo

**Nhac nen:** Guitar nhanh, vui — di chuyen theo nhip nhac

---

### Che do 2: HAPPY_LOW (Vui + Nhe nhang)

**Cam xuc:** Nhu dang nhin hoang hon dep, thu vi nhe nhang.

**Lam gi:**
- Tay o vung giua man hinh
- Di chuyen CHAM — khoang 1 lan/2 giay
- Bien do VUA — qua lai ~30cm
- Chuyen dong tron, muot ma

**Nhac nen:** Guitar cham, nhe — di chuyen theo song am thanh

---

### Che do 3: SAD_HIGH (Buon + Cang thang)

**Cam xuc:** Nhu dang bop chat tay vi tuc gian hoac lo lang.

**Lam gi:**
- Tay de THAP (vung duoi man hinh)
- Di chuyen GAP va khong deu — co rut lai roi day ra
- Chuyen dong nho, khong rong
- Co the hoi run nhe

**Nhac nen:** Strings tram, can thang

---

### Che do 4: SAD_LOW (Buon + Cham)

**Cam xuc:** Nhu dang met moi, keo le buoc chan.

**Lam gi:**
- Tay de THAP NHAT trong khung hinh
- Di chuyen RAT CHAM — khoang 1 lan/4 giay
- Nhu tay rat nang, khong muon nhac len
- Chuyen dong thang, it bien do

**Nhac nen:** Strings rat cham, buon

---

### Che do 5: NEUTRAL (Trung tinh)

**Cam xuc:** Bình thuong, khong co cam xuc dac biet.

**Lam gi:**
- Tay o GIUA man hinh
- Di chuyen DEU DANG — khong qua nhanh, khong qua cham
- Nhu dang go ban phim hoac lam viec binh thuong
- Khong co bieu cam dac biet

**Nhac nen:** Piano, nhip dieu

---

### Che do 6: NONE (Khong co tay)

**Lam gi:**
- Ha tay xuong, ra ngoai khung hinh
- Ngoi yen, khong lam gi ca
- Khung hinh hien thi nua than tren nhung KHONG thay tay

---

## Quy trinh thu moi che do

```
1. Script hien man hinh gioi thieu che do
2. Doc ky ten che do va huong dan
3. Nhan SPACE
4. Dem nguoc 3-2-1
5. Bat nhac nen
6. Di chuyen tay THEO CAM XUC trong khi nghe nhac
7. Thanh tien do day -> tu dong chuyen che do tiep
8. Nhan SPACE de sang che do tiep
```

---

## Meo de thu tot

1. **Nghe nhac truoc** khi bat dau di chuyen — de cam xuc tu nhien
2. **Dung dien xuat** — lam qua thi cung tot hon lam it
3. **Da dang chuyen dong** trong cung 1 che do — dung phat lai dung 1 dong tac
4. **Doi chieu sang** neu MediaPipe khong nhan dien duoc tay

---

## Upload sau khi thu xong

Sau khi thu xong tat ca 6 che do:
1. Tim file: `data/raw/gesture_data_v2.csv`
2. Upload len Google Drive
3. Doi ten: `gesture_data_v2_[ten_ban].csv`
4. Nhan link cho nhom truong
