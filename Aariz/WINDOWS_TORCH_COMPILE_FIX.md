# ⚠️ رفع مشکل torch.compile روی Windows

## مشکل

`torch.compile` نیاز به **Triton** دارد که روی Windows به صورت پیش‌فرض نصب نیست و ممکن است مشکل داشته باشد.

**خطا:**
```
RuntimeError: Cannot find a working triton installation
```

---

## ✅ راهکار اعمال شده

### 1. غیرفعال کردن torch.compile روی Windows
- `torch.compile` از script حذف شد
- کد به صورت خودکار بررسی می‌کند که آیا Triton نصب است یا نه
- اگر نصب نباشد، بدون خطا ادامه می‌دهد

### 2. کاهش num_workers (8 → 6)
- سیستم شما فقط 6 worker پیشنهاد می‌دهد
- `num_workers` به 6 کاهش یافت

### 3. فعال نگه داشتن channels_last
- `channels_last` همچنان فعال است (10-20% speedup)
- این بهینه‌سازی نیاز به Triton ندارد

---

## 🚀 بهینه‌سازی‌های فعال

### فعال:
- ✅ **channels_last**: 10-20% speedup (بدون نیاز به Triton)
- ✅ **Mixed Precision (FP16)**: 2x speedup
- ✅ **Early Stopping**: صرفه‌جویی در زمان
- ✅ **num_workers=6**: بهینه برای Windows

### غیرفعال (به دلیل Windows):
- ❌ **torch.compile**: نیاز به Triton (مشکل روی Windows)

---

## 📊 نتایج انتظاری

### بدون torch.compile:
- Epoch اول: ~8-12 دقیقه
- Epoch‌های بعدی: ~5-7 دقیقه
- افزایش سرعت: 10-20% (از channels_last)
- CPU Utilization: 70-80% (با num_workers=6)

### اگر می‌خواهید torch.compile را فعال کنید:

#### گزینه 1: نصب Triton (ممکن است مشکل داشته باشد)
```bash
pip install triton
```

#### گزینه 2: استفاده از fallback mode
```python
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```

**نکته:** Triton روی Windows ممکن است مشکل داشته باشد. بهتر است از `channels_last` استفاده کنید.

---

## 🎯 دستور نهایی (بدون torch.compile)

```bash
python train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 200 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --mixed_precision \
    --use_ema \
    --multi_gpu \
    --num_workers 6 \
    --channels_last \
    --early_stopping \
    --patience 20 \
    --save_frequency 5
```

یا استفاده از script:
```bash
train_1024x1024.bat
```

---

## 💡 نکات مهم

1. **torch.compile روی Windows**: معمولاً مشکل دارد، بهتر است غیرفعال باشد
2. **channels_last**: بهترین بهینه‌سازی برای Windows (بدون نیاز به Triton)
3. **num_workers=6**: بهینه برای سیستم شما
4. **Mixed Precision**: همچنان فعال است و سرعت را افزایش می‌دهد

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ مشکل torch.compile روی Windows حل شد
















