# راهنمای بهبودهای اعمال شده در Training

## ✅ تغییرات اعمال شده

### 1. Adaptive Wing Loss
- ✅ اضافه شده به `train.py`
- ✅ Loss function بهینه‌تر برای landmark detection
- ✅ بهتر از MSE/Focal Loss برای دقت بالا

### 2. WarmupCosineScheduler
- ✅ اضافه شده به `train.py`
- ✅ Warmup برای 5 epoch اول
- ✅ Cosine Annealing with Restarts برای بهبود convergence

### 3. Learning Rate
- ✅ Default LR تغییر کرده: `1e-4` → `5e-4`
- ✅ با Warmup به تدریج افزایش می‌یابد

## 🚀 استفاده

### استفاده با تنظیمات جدید (پیشنهادی):

```bash
# شروع جدید با Adaptive Wing Loss
python train.py \
    --dataset_path Aariz \
    --model resnet \
    --batch_size 8 \
    --epochs 100 \
    --lr 5e-4 \
    --loss adaptive_wing \
    --warmup_epochs 5
```

### ادامه از checkpoint قبلی:

```bash
# Fine-tuning از بهترین checkpoint
python train.py \
    --resume checkpoints/checkpoint_best.pth \
    --dataset_path Aariz \
    --model resnet \
    --lr 2e-4 \
    --loss adaptive_wing \
    --warmup_epochs 3 \
    --epochs 50
```

### استفاده از Loss قبلی (اگر می‌خواهید):

```bash
# اگر می‌خواهید از HeatmapLoss استفاده کنید
python train.py \
    --loss heatmap \
    --lr 1e-4
```

## 📊 پارامترهای جدید

| پارامتر | مقدار پیش‌فرض | توضیح |
|---------|---------------|-------|
| `--loss` | `adaptive_wing` | نوع loss function |
| `--lr` | `5e-4` | Learning rate اولیه |
| `--warmup_epochs` | `5` | تعداد epoch های warmup |

## 🎯 انتظارات

با تغییرات جدید:
- **MRE**: باید به 3-4mm در 50 epoch برسد
- **SDR @ 2mm**: باید به 30-40% برسد
- **Convergence**: سریع‌تر و پایدارتر

## ⚠️ نکات مهم

1. **اگر از checkpoint قبلی استفاده می‌کنید:**
   - Learning rate را کاهش دهید (`2e-4` یا `1e-4`)
   - Warmup را کوتاه کنید (`3` epoch)

2. **اگر آموزش از اول شروع می‌کنید:**
   - از تنظیمات پیش‌فرض استفاده کنید
   - `--loss adaptive_wing` به صورت خودکار استفاده می‌شود

3. **مانیتورینگ:**
   - TensorBoard را بررسی کنید: `tensorboard --logdir logs`
   - Learning Rate را در output ببینید

## 📈 مقایسه

| Loss Function | انتظار MRE | زمان آموزش |
|---------------|------------|------------|
| HeatmapLoss (قدیمی) | ~5mm | Baseline |
| Adaptive Wing Loss (جدید) | ~3-4mm | مشابه |

---

**موفق باشید! 🚀**

