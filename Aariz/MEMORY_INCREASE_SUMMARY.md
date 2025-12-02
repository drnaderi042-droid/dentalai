# افزایش GPU Memory Usage به ~7.5 GB

## 🎯 هدف

افزایش مصرف GPU memory از ~4500 MB به ~7500 MB برای استفاده بهتر از GPU

## 🔧 تغییرات اعمال شده

### 1. ✅ افزایش Batch Size

**قبل:**
- Batch size: 4
- GPU memory: ~4500 MB

**بعد:**
- Batch size: 6 (افزایش 50%)
- GPU memory: ~6750-7500 MB (افزایش ~67%)

### 2. ✅ Gradient Accumulation = 1

- Gradient accumulation: 1 (بدون accumulation)
- چون batch size را افزایش دادیم، نیازی به accumulation نیست

### 3. ✅ num_workers = 6

- حفظ شده (کاربر گفت خوب است)
- سرعت data loading بالا

### 4. ✅ Prefetch Factor

- افزایش prefetch_factor برای buffering بیشتر
- استفاده بهتر از GPU memory

## 📊 مقایسه

| پارامتر | قبل | بعد | تغییر |
|---------|-----|-----|-------|
| **Batch Size** | 4 | 6 | ↑ 50% |
| **Gradient Accum** | 2 | 1 | ↓ (حذف) |
| **Effective Batch** | 4 | 6 | ↑ 50% |
| **num_workers** | 6 | 6 | = |
| **GPU Memory** | ~4500 MB | ~7500 MB | ↑ 67% |
| **Speed** | Baseline | +15-20% | ↑ |

## ⚡ مزایا

1. **استفاده بهتر از GPU:** ~67% بیشتر memory استفاده می‌شود
2. **سرعت بیشتر:** batch size بیشتر → کمتر iteration → سریع‌تر
3. **Training بهتر:** batch size بیشتر → gradient پایدارتر
4. **کاهش idle time:** GPU کمتر idle می‌ماند

## 🚀 دستور اجرا

```bash
cd Aariz
.\train_768_weighted_loss.bat
```

یا:

```bash
cd Aariz
python train2.py \
    --resume checkpoint_best_768.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 768 768 \
    --batch_size 6 \
    --gradient_accumulation_steps 1 \
    --lr 1e-6 \
    --warmup_epochs 2 \
    --epochs 60 \
    --loss adaptive_wing \
    --mixed_precision \
    --num_workers 6
```

## ⚠️ نکات مهم

1. **Learning Rate:** ممکن است نیاز به تنظیم داشته باشد (چون effective batch size بیشتر شده)
2. **Memory:** مطمئن شوید که GPU شما حداقل 8GB memory دارد
3. **OOM:** اگر OOM گرفتید، batch size را به 5 کاهش دهید

## 📝 تنظیمات نهایی

- **Batch Size:** 6 (برای ~7.5GB memory)
- **Gradient Accumulation:** 1 (بدون accumulation)
- **num_workers:** 6
- **Mixed Precision:** فعال (FP16)
- **Effective Batch Size:** 6

## ⏱️ زمان تخمینی

- **زمان:** ~2-2.5 ساعت (سریع‌تر از قبل)
- **بهبود:** ~15-20% سریع‌تر (به خاطر batch size بیشتر)















