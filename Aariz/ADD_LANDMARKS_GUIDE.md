# راهنمای اضافه کردن لندمارک‌های جدید به مدل Aariz

## 📋 خلاصه

این راهنما نحوه اضافه کردن لندمارک‌های جدید (مثل PT - Pterygoid) به مدل 512x512 موجود را توضیح می‌دهد.

## 🎯 مراحل اضافه کردن لندمارک‌های جدید

### مرحله 1: آنوتیت کردن تصاویر با لندمارک‌های جدید

ابتدا باید تصاویر دیتاست را با لندمارک‌های جدید آنوتیت کنید:

1. **استفاده از ابزار آنوتیت** (مثل LabelMe، CVAT، یا ابزار سفارشی)
2. **فرمت آنوتیت**: JSON با فرمت Aariz
3. **نمونه آنوتیت برای لندمارک PT**:
```json
{
  "landmarks": [
    {
      "symbol": "PT",
      "value": {
        "x": 1234.5,
        "y": 567.8
      }
    }
  ]
}
```

### مرحله 2: آماده‌سازی دیتاست

لندمارک‌های جدید را به فایل‌های JSON آنوتیت اضافه کنید. فایل‌های آنوتیت در مسیر زیر هستند:
```
Aariz/
├── train/Annotations/Cephalometric Landmarks/Senior Orthodontists/
├── valid/Annotations/Cephalometric Landmarks/Senior Orthodontists/
└── test/Annotations/Cephalometric Landmarks/Senior Orthodontists/
```

### مرحله 3: Fine-tuning مدل

از اسکریپت `finetune_extended_landmarks.py` استفاده کنید:

```bash
python finetune_extended_landmarks.py \
    --checkpoint checkpoints/checkpoint_best_512x512.pth \
    --additional_landmarks PT PTL PTR \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 512 512 \
    --save_dir checkpoints_extended \
    --log_dir logs_extended \
    --model hrnet \
    --mixed_precision
```

### پارامترهای مهم:

- `--checkpoint`: مسیر checkpoint مدل 512x512 موجود
- `--additional_landmarks`: لیست لندمارک‌های جدید (مثلاً PT PTL PTR)
- `--batch_size`: اندازه batch (برای RTX 3070 Ti: 4-6)
- `--epochs`: تعداد epoch‌ها (پیشنهاد: 50-100)
- `--lr`: Learning rate (پیشنهاد: 1e-4)
- `--model`: معماری مدل (hrnet, resnet, unet)

## 🔧 نحوه کار

### 1. Extended Dataset (`dataset_extended.py`)

- لندمارک‌های پایه (29 عدد) + لندمارک‌های جدید
- پشتیبانی از لندمارک‌های اختیاری (اگر در آنوتیت نباشند، -1 می‌شوند)

### 2. Extended Model (`finetune_extended_landmarks.py`)

- **Transfer Learning**: Backbone مدل فریز می‌شود
- فقط head جدید برای لندمارک‌های اضافی آموزش می‌بیند
- کاهش زمان آموزش و نیاز به داده کمتر

### 3. ساختار مدل Extended

```
Base Model (29 landmarks) [Frozen]
    ↓
Features
    ↓
New Head (N new landmarks) [Trainable]
    ↓
Concatenate → Total (29 + N landmarks)
```

## 📝 مثال: اضافه کردن لندمارک PT (Pterygoid)

### 1. تعریف لندمارک‌های جدید

```python
additional_landmarks = ["PT"]  # یا ["PT", "PTL", "PTR"] برای چند لندمارک
```

### 2. ایجاد Dataset

```python
from dataset_extended import ExtendedAarizDataset

dataset = ExtendedAarizDataset(
    "Aariz",
    mode="TRAIN",
    additional_landmarks=["PT"]
)
```

### 3. Fine-tuning

```bash
python finetune_extended_landmarks.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --additional_landmarks PT \
    --epochs 50 \
    --lr 1e-4
```

## ⚠️ نکات مهم

1. **آنوتیت کامل**: حداقل 50-100 تصویر با لندمارک‌های جدید آنوتیت کنید
2. **توزیع داده**: لندمارک‌های جدید را در train/valid/test توزیع کنید
3. **Learning Rate**: برای fine-tuning از LR کوچک‌تر استفاده کنید (1e-4 تا 1e-5)
4. **Freeze Backbone**: Backbone فریز می‌شود تا weights موجود حفظ شود
5. **Validation**: حتماً validation set داشته باشید

## 📊 لندمارک‌های پیشنهادی برای اضافه کردن

بر اساس نیازهای کلینیکی، این لندمارک‌ها می‌توانند مفید باشند:

- **PT** - Pterygoid (پتریگوئید)
- **PTL** - Pterygoid Left
- **PTR** - Pterygoid Right
- **Ba** - Basion
- **Cd** - Condylion (ممکن است با Co متفاوت باشد)
- **UIE** - Upper Incisor Edge
- **LIE** - Lower Incisor Edge

## 🔍 بررسی نتایج

پس از آموزش، می‌توانید نتایج را بررسی کنید:

```python
from utils import load_checkpoint
import torch

checkpoint = torch.load('checkpoints_extended/checkpoint_best.pth')
print(f"Number of landmarks: {checkpoint['num_landmarks']}")
print(f"Additional landmarks: {checkpoint['additional_landmarks']}")
print(f"MRE: {checkpoint['mre']:.2f}mm")
print(f"SDR@2mm: {checkpoint['sdr_2mm']:.2f}%")
```

## 📁 فایل‌های ایجاد شده

1. `dataset_extended.py` - Dataset با پشتیبانی از لندمارک‌های اضافی
2. `finetune_extended_landmarks.py` - اسکریپت fine-tuning
3. `finetune_extended_landmarks.bat` - فایل batch برای اجرای آسان
4. `ADD_LANDMARKS_GUIDE.md` - این راهنما

## 🚀 یادآوری تست CLdetection2023

برای تست دقت مدل CLdetection2023 روی لندمارک‌های مشترک:

```bash
cd Aariz
python test_cldetection_final.py
```

(نیاز به نصب MMPose در محیط conda دارد)
















