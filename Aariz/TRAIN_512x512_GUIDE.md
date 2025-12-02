# 🎯 راهنمای Training با Image Size 512×512

## 📊 چرا 512×512؟

- ✅ **دقت بالاتر**: Resolution بیشتر = جزئیات بیشتر
- ✅ **نتایج بهتر**: معمولاً 3-7% بهبود SDR
- ⚠️ **نیاز به VRAM بیشتر**: حداقل 8GB GPU
- ⚠️ **زمان بیشتر**: ~2x زمان training نسبت به 256×256

---

## 🔧 تنظیمات پیشنهادی

### برای RTX 3070 Ti (8GB VRAM)

```bash
python train2.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 512 512 \
    --batch_size 8 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --epochs 100 \
    --loss adaptive_wing \
    --mixed_precision
```

**پارامترهای کلیدی:**
- `--image_size 512 512`: سایز جدید ✅
- `--batch_size 8`: کاهش از 16 (به خاطر VRAM)
- `--mixed_precision`: برای کاهش VRAM استفاده

---

## 🎯 دو استراتژی

### استراتژی 1: Fine-tuning از Checkpoint فعلی ⭐ (پیشنهادی)

**مزایا:**
- ✅ سریع‌تر (50 epoch کافی است)
- ✅ از یادگیری قبلی استفاده می‌کند
- ✅ کم‌ریسک‌تر

**دستور:**
```bash
python train2.py \
    --resume checkpoints/checkpoint_best.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 512 512 \
    --batch_size 8 \
    --lr 1e-5 \
    --warmup_epochs 3 \
    --epochs 50 \
    --loss adaptive_wing \
    --mixed_precision
```

**نکات:**
- `--lr 1e-5`: Learning rate پایین (fine-tuning)
- `--epochs 50`: کافی است برای fine-tuning

**زمان**: 4-6 ساعت (با RTX 3070 Ti)

---

### استراتژی 2: آموزش از اول

**زمان**: 8-12 ساعت (100 epochs)

**دستور:**
```bash
python train2.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 512 512 \
    --batch_size 8 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --epochs 100 \
    --loss adaptive_wing \
    --mixed_precision
```

---

## 📋 استفاده از Batch File (ساده‌ترین)

```batch
Aariz\train_512x512.bat
```

این batch file به صورت خودکار:
- بررسی می‌کند checkpoint موجود است یا نه
- اگر موجود باشد: گزینه fine-tuning می‌دهد
- اگر نباشد: از اول شروع می‌کند

---

## ⚙️ تنظیمات پیشرفته

### اگر Out of Memory Error گرفتید:

**گزینه 1: کاهش Batch Size**
```bash
--batch_size 6  # یا 4
```

**گزینه 2: استفاده از Gradient Accumulation**
(نیاز به تغییر کد train2.py)

**گزینه 3: استفاده از Image Size کوچک‌تر**
```bash
--image_size 384 384  # تعادل بین دقت و VRAM
```

---

## 📊 نتایج انتظاری

### Fine-tuning (از 256×256 checkpoint):
- **MRE**: از 1.99mm به **1.5-1.7mm**
- **SDR @ 2mm**: از 65% به **70-75%**
- **زمان**: 4-6 ساعت

### Training از اول:
- **MRE**: **1.3-1.6mm**
- **SDR @ 2mm**: **72-78%**
- **زمان**: 8-12 ساعت

---

## 🔍 رصد کردن Training

### Tensorboard:
```bash
tensorboard --logdir logs
```

### بررسی Checkpoints:
```bash
# هر 10 epoch یک checkpoint ذخیره می‌شود
ls checkpoints/

# پیدا کردن بهترین checkpoint:
python find_best_checkpoint.py checkpoints/
```

---

## ⚠️ نکات مهم

1. **VRAM**: حداقل 8GB لازم است
2. **Heatmap Sigma**: باید متناسب با image size باشد
   - برای 512×512: sigma ≈ 6.0 (2x بیشتر از 256×256)
3. **Batch Size**: با 512×512، batch_size=8 حداکثر برای 8GB VRAM
4. **زمان**: ~2x زمان بیشتر نسبت به 256×256

---

## 🚀 دستورات سریع

### Fine-tuning (پیشنهادی):
```bash
cd Aariz
python train2.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 512 512 --batch_size 8 --lr 1e-5 --warmup_epochs 3 --epochs 50 --loss adaptive_wing --mixed_precision
```

### یا استفاده از Batch File:
```batch
Aariz\train_512x512.bat
```

---

## 📈 مراحل بعدی

بعد از training:
1. ✅ تست روی validation set
2. ✅ مقایسه با نتایج قبلی (256×256)
3. ✅ استفاده از TTA برای بهبود بیشتر (اختیاری)

---

**تاریخ**: 2024-11-01
**وضعیت**: ✅ آماده برای استفاده

