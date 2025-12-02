# 🔧 Training Fix Notes

## ⚠️ مشکلات قبلی:

### 1. **Pixel Error خیلی بالا** (166-240 px)
**علت:** محاسبه اشتباه pixel error  
**قبل:**
```python
errors = np.sqrt(np.sum((pred_pixels - gt_pixels) ** 2, axis=1))
```
این فاصله اقلیدسی برای همه 4 مختصات (p1_x, p1_y, p2_x, p2_y) را **با هم** می‌گیرد!

**بعد (اصلاح شده):** ✅
```python
# محاسبه جداگانه برای p1 و p2
p1_errors = np.sqrt((pred_pixels[:, 0] - gt_pixels[:, 0])**2 + 
                   (pred_pixels[:, 1] - gt_pixels[:, 1])**2)
p2_errors = np.sqrt((pred_pixels[:, 2] - gt_pixels[:, 2])**2 + 
                   (pred_pixels[:, 3] - gt_pixels[:, 3])**2)
avg_errors = (p1_errors + p2_errors) / 2
```

### 2. **Learning Rate کم**
**قبل:** `0.001`  
**بعد:** `0.003` ✅ (3x سریعتر)

### 3. **Early Stopping خیلی کند**
**قبل:**
- Scheduler patience: 20
- Early stop patience: 50

**بعد:** ✅
- Scheduler patience: 10 (2x سریعتر)
- Early stop patience: 30 (صرفه‌جویی وقت)

---

## 🎯 تنظیمات نهایی برای 768px

```python
hrnet_variant='hrnet_w18'
image_size=768
batch_size=2
num_epochs=200
learning_rate=0.003          # Increased! ⬆️
scheduler_patience=10         # Faster LR decay ⚡
early_stop_patience=30        # Faster stopping ⚡
```

---

## 📊 انتظارات جدید

| Metric | قبل (اشتباه) | بعد (درست) |
|--------|--------------|-------------|
| **Pixel Error** | 166-240 px ❌ | 20-40 px ✅ |
| **Convergence** | کند | سریع‌تر ⚡ |
| **Validation Loss** | ~0.01-0.02 | ~0.003-0.008 |

---

## 🚀 چگونه دوباره Train کنیم؟

### گام 1: متوقف کردن training قبلی

```
Ctrl + C
```

### گام 2: پاک کردن مدل قدیمی (اختیاری)

```cmd
del models\hrnet_p1p2_best_hrnet_w18.pth
```

یا rename:
```cmd
ren models\hrnet_p1p2_best_hrnet_w18.pth hrnet_p1p2_old.pth
```

### گام 3: شروع training جدید

```cmd
cd aariz
train_hrnet_768.bat
```

یا:

```cmd
python train_p1_p2_hrnet.py
```

---

## 📈 نظارت بر Progress

### چیزهای که باید ببینید:

✅ **Epoch 1-10:** Train loss کاهش سریع (0.06 → 0.005)  
✅ **Epoch 10-30:** Pixel error کاهش (باید به زیر 50px برسد)  
✅ **Epoch 30-100:** Stabilization (pixel error: 20-40px)

### علائم خوب:

```
Epoch 20/200:
  Train Loss: 0.003215
  Val Loss: 0.005124
  Avg Pixel Error: 35.47 px   ← خیلی بهتر!
  Learning Rate: 0.003000
  >>> Best model saved!
```

### علائم بد:

```
Epoch 20/200:
  Train Loss: 0.005000
  Val Loss: 0.020000
  Avg Pixel Error: 180.00 px  ← هنوز بالاست!
```

اگر این را دیدید:
1. ✅ مطمئن شوید `annotations_p1_p2.json` درست است
2. ✅ بررسی کنید که GPU استفاده می‌شود
3. ✅ تصاویر را با `check_annotations_quality.py` بررسی کنید

---

## 🧪 بعد از Training

```cmd
cd aariz
test_hrnet.bat
```

### انتظارات:

```
[RESULTS] Test Statistics:
  - Samples tested: 100
  - Average error: 25.43 px    ← خیلی بهتر!
  - Median error: 22.15 px
  - Min error: 8.52 px
  - Max error: 45.89 px
```

---

## ❓ سوالات متداول

### Q1: Training خیلی کنده؟
**A:** با `batch_size=2` و `768px`، هر epoch ~40-50 ثانیه طول می‌کشد. این نرمال است.

### Q2: Pixel error هنوز بالاست؟
**A:** بررسی کنید:
```cmd
python check_annotations_quality.py annotations_p1_p2.json
```

اگر annotations مشکل دارند، دوباره annotate کنید.

### Q3: چه موقع متوقف کنم؟
**A:** وقتی که:
- Pixel error < 30px (خوب)
- Pixel error < 20px (عالی)
- Early stopping خودکار متوقف کند

---

## 📝 خلاصه تغییرات

1. ✅ محاسبه pixel error اصلاح شد
2. ✅ Learning rate 3x شد (0.001 → 0.003)
3. ✅ Scheduler patience کاهش یافت (20 → 10)
4. ✅ Early stopping سریع‌تر (50 → 30)
5. ✅ Image size: 768px (accuracy بالاتر)

---

**شروع training جدید:**

```cmd
cd aariz
train_hrnet_768.bat
```

**زمان تخمینی:** 2-4 ساعت (سریع‌تر از قبل!)

**موفق باشید! 🚀**













