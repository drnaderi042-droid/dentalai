# راهنمای Fine-tuning مدل با دیتاست جدید 3471833

## 📋 خلاصه

این راهنما نحوه fine-tuning مدل آموزش‌دیده با دیتاست Aariz (29 لندمارک) را با استفاده از دیتاست جدید 3471833 (19 لندمارک) توضیح می‌دهد.

## 🎯 هدف

- بهبود دقت مدل با استفاده از دیتاست جدید
- استفاده از دانش آموخته شده از دیتاست Aariz
- افزایش دقت تشخیص لندمارک‌ها

## 📁 فایل‌های ایجاد شده

1. **`dataset_3471833.py`**: Dataset loader برای دیتاست 3471833 با 19 لندمارک
2. **`dataset_combined.py`**: Dataset ترکیبی که از هر دو دیتاست استفاده می‌کند
3. **`finetune_combined.py`**: اسکریپت fine-tuning اصلی
4. **`finetune_combined.bat`**: فایل batch برای اجرای آسان در Windows

## 🔧 پیش‌نیازها

1. مدل آموزش‌دیده با دیتاست Aariz (checkpoint)
2. دیتاست 3471833 در پوشه `3471833`
3. دیتاست Aariz در پوشه `Aariz`

## 🚀 نحوه استفاده

### روش 1: استفاده از فایل Batch (پیشنهادی)

```bash
cd Aariz
finetune_combined.bat
```

### روش 2: اجرای مستقیم Python

```bash
cd Aariz
python finetune_combined.py \
    --model hrnet \
    --resume checkpoints/checkpoint_best.pth \
    --aariz_path Aariz \
    --dataset_3471833_path ../3471833 \
    --batch_size 6 \
    --epochs 50 \
    --lr 1e-5 \
    --image_size 512 512 \
    --loss adaptive_wing \
    --mixed_precision
```

## ⚙️ پارامترهای مهم

### پارامترهای اصلی

- `--resume`: مسیر checkpoint مدل آموزش‌دیده با Aariz (الزامی)
- `--aariz_path`: مسیر دیتاست Aariz (پیش‌فرض: `Aariz`)
- `--dataset_3471833_path`: مسیر دیتاست 3471833 (پیش‌فرض: `../3471833`)
- `--model`: معماری مدل (`hrnet`, `resnet`, `unet`, `hourglass`)
- `--batch_size`: اندازه batch (پیش‌فرض: 6)
- `--epochs`: تعداد epochs (پیش‌فرض: 50)
- `--lr`: Learning rate (پیش‌فرض: 1e-5 برای fine-tuning)

### پارامترهای پیشرفته

- `--use_aariz_only`: فقط از دیتاست Aariz استفاده کند
- `--use_3471833_only`: فقط از دیتاست 3471833 استفاده کند
- `--freeze_backbone`: فریز کردن backbone (فقط head آموزش داده می‌شود)
- `--mixed_precision`: استفاده از FP16 برای سرعت بیشتر

## 📊 Mapping لندمارک‌ها

دیتاست 3471833 دارای 19 لندمارک است که به 29 لندمارک دیتاست Aariz نگاشت می‌شوند:

| Index (19) | نام لندمارک | Index (29) | نام در Aariz |
|------------|-------------|-------------|--------------|
| 0 | S (Sella) | 10 | S |
| 1 | N (Nasion) | 4 | N |
| 2 | Or (Orbitale) | 5 | Or |
| 3 | A (Subspinale) | 0 | A |
| 4 | B (Supramentale) | 2 | B |
| 5 | PNS | 7 | PNS |
| 6 | ANS | 1 | ANS |
| 7 | U1 | 20 | UPM |
| 8 | L1 | 23 | LIA |
| 9 | Me (Menton) | 13 | Me |
| 10 | U6 | 19 | UPM |
| 11 | L6 | 22 | LMT |
| 12 | Go (Gonion) | 14 | Go |
| 13 | Pog (Pogonion) | 6 | Pog |
| 14 | Gn (Gnathion) | 12 | Gn |
| 15 | Ar (Articulare) | 11 | Ar |
| 16 | Co (Condylion) | 12 | Co |
| 17 | Po (Porion) | 9 | Po |
| 18 | R (Ramus) | 8 | R |

## 📈 استراتژی Fine-tuning

### مرحله 1: Fine-tuning با Learning Rate پایین

```bash
python finetune_combined.py \
    --resume checkpoints/checkpoint_best.pth \
    --lr 1e-5 \
    --epochs 30 \
    --mixed_precision
```

### مرحله 2: ادامه با Learning Rate بالاتر (اختیاری)

اگر نتایج خوب نبود، می‌توانید با LR بالاتر ادامه دهید:

```bash
python finetune_combined.py \
    --resume checkpoints_finetuned/checkpoint_best.pth \
    --lr 5e-5 \
    --epochs 20 \
    --mixed_precision
```

### مرحله 3: Fine-tuning فقط با دیتاست جدید (اختیاری)

برای تمرکز بیشتر روی دیتاست جدید:

```bash
python finetune_combined.py \
    --resume checkpoints/checkpoint_best.pth \
    --use_3471833_only \
    --lr 1e-5 \
    --epochs 50
```

## 📁 ساختار خروجی

پس از fine-tuning، فایل‌های زیر ایجاد می‌شوند:

```
checkpoints_finetuned/
├── checkpoint_best.pth      # بهترین مدل (کمترین MRE)
├── checkpoint_latest.pth     # آخرین checkpoint
└── checkpoint_epoch_*.pth    # Checkpoint های دوره‌ای

logs_finetuned/
└── hrnet_finetuned_YYYYMMDD_HHMMSS/  # لاگ‌های TensorBoard
```

## 🔍 نظارت بر آموزش

برای مشاهده نمودارهای TensorBoard:

```bash
tensorboard --logdir logs_finetuned
```

سپس به `http://localhost:6006` بروید.

## 💡 نکات مهم

1. **Learning Rate**: برای fine-tuning از LR پایین (1e-5 تا 5e-5) استفاده کنید
2. **Batch Size**: با توجه به VRAM GPU خود تنظیم کنید
3. **Epochs**: معمولاً 30-50 epoch برای fine-tuning کافی است
4. **Mixed Precision**: برای سرعت بیشتر و مصرف کمتر VRAM فعال کنید
5. **Freeze Backbone**: اگر می‌خواهید فقط head را آموزش دهید، از `--freeze_backbone` استفاده کنید

## 🐛 عیب‌یابی

### مشکل: Out of Memory (OOM)

- کاهش `batch_size`
- کاهش `image_size`
- استفاده از `--mixed_precision`
- استفاده از `--freeze_backbone`

### مشکل: دقت بهبود نمی‌یابد

- کاهش `--lr` (مثلاً 5e-6)
- افزایش `--epochs`
- بررسی mapping لندمارک‌ها
- استفاده از `--use_3471833_only` برای تمرکز روی دیتاست جدید

### مشکل: Dataset پیدا نمی‌شود

- بررسی مسیرهای `--aariz_path` و `--dataset_3471833_path`
- اطمینان از وجود فایل‌های annotation و image

## 📝 مثال کامل

```bash
# Fine-tuning با تنظیمات بهینه
python finetune_combined.py \
    --model hrnet \
    --resume checkpoints/checkpoint_best.pth \
    --aariz_path Aariz \
    --dataset_3471833_path ../3471833 \
    --batch_size 6 \
    --epochs 50 \
    --lr 1e-5 \
    --image_size 512 512 \
    --loss adaptive_wing \
    --aariz_annotation_type "Senior Orthodontists" \
    --dataset_3471833_annotation_type "400_senior" \
    --mixed_precision
```

## ✅ بررسی نتایج

پس از اتمام fine-tuning، می‌توانید نتایج را بررسی کنید:

```python
from inference import load_model, predict_landmarks

# بارگذاری مدل fine-tuned
model = load_model('checkpoints_finetuned/checkpoint_best.pth')

# تست روی یک تصویر
landmarks = predict_landmarks(model, 'path/to/image.png')
```

## 📚 منابع بیشتر

- برای اطلاعات بیشتر درباره آموزش اولیه: `README_FA.md`
- برای راهنمای HRNet: `HRNET_GUIDE.md`
- برای بهینه‌سازی VRAM: `VRAM_OPTIMIZATION.md`
















