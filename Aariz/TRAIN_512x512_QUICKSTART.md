# ⚡ Quick Start: Training با 512×512

## 🚀 سریع‌ترین روش

### استفاده از Batch File (توصیه می‌شود):
```batch
Aariz\train_512x512.bat
```

این batch file:
- ✅ به صورت خودکار checkpoint را بررسی می‌کند
- ✅ گزینه fine-tuning یا از اول را می‌دهد
- ✅ تنظیمات بهینه را اعمال می‌کند

---

## 📝 دستورات دستی

### گزینه 1: Fine-tuning (پیشنهادی) - 4-6 ساعت

```powershell
cd Aariz
python train2.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 512 512 --batch_size 8 --lr 1e-5 --warmup_epochs 3 --epochs 50 --loss adaptive_wing --mixed_precision
```

### گزینه 2: از اول - 8-12 ساعت

```powershell
cd Aariz
python train2.py --dataset_path Aariz --model hrnet --image_size 512 512 --batch_size 8 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision
```

---

## ⚙️ تنظیمات

| پارامتر | مقدار | توضیح |
|---------|-------|-------|
| `--image_size` | 512 512 | ✅ سایز جدید |
| `--batch_size` | 8 | برای RTX 3070 Ti (8GB) |
| `--lr` | 1e-5 (fine-tuning) یا 5e-4 (از اول) | Learning rate |
| `--epochs` | 50 (fine-tuning) یا 100 (از اول) | تعداد epochs |
| `--loss` | adaptive_wing | بهترین loss function |
| `--mixed_precision` | - | کاهش VRAM استفاده |

---

## 📊 نتایج انتظاری

### Fine-tuning:
- **MRE**: 1.5-1.7mm (بهبود از 1.99mm)
- **SDR @ 2mm**: 70-75% (بهبود از 65%)
- **زمان**: 4-6 ساعت

### از اول:
- **MRE**: 1.3-1.6mm
- **SDR @ 2mm**: 72-78%
- **زمان**: 8-12 ساعت

---

## ⚠️ نکات مهم

1. **VRAM**: حداقل 8GB لازم است
2. **Heatmap Sigma**: خودکار تنظیم می‌شود (≈6.0 برای 512×512)
3. **اگر Out of Memory**: `--batch_size` را به 6 یا 4 کاهش دهید

---

## 🔍 رصد کردن

```powershell
# Tensorboard
tensorboard --logdir logs
```

---

**شروع کنید با:** `Aariz\train_512x512.bat` 🚀

