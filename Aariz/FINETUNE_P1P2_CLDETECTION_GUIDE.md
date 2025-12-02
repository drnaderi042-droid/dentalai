# راهنمای Fine-tuning مدل CLdetection2023 برای P1/P2

## 📋 خلاصه

این اسکریپت مدل CLdetection2023 که قبلاً برای 19 لندمارک آموزش دیده را fine-tune می‌کند تا 2 لندمارک جدید (p1, p2) را نیز تشخیص دهد.

## 🎯 مزایای استفاده از مدل CLdetection2023

1. **Backbone قوی**: مدل CLdetection2023 روی دیتاست بزرگ آموزش دیده و features قوی دارد
2. **Transfer Learning**: استفاده از دانش قبلی برای یادگیری سریع‌تر
3. **دقت بالاتر**: معمولاً بهتر از آموزش از صفر عمل می‌کند

## ⚠️ مشکلات احتمالی دیتاست‌های متفاوت

### 1. **Normalization متفاوت**
- **مشکل**: CLdetection2023 از normalization خاص خود استفاده می‌کند
- **راه‌حل**: اسکریپت از normalization صحیح CLdetection2023 استفاده می‌کند:
  - mean=[121.25, 121.25, 121.25]
  - std=[76.5, 76.5, 76.5]

### 2. **Image Size متفاوت**
- **مشکل**: مدل CLdetection2023 روی سایز 1024x1024 آموزش دیده
- **راه‌حل**: اسکریپت همه تصاویر را به 1024x1024 resize می‌کند (مطابق با CLdetection2023)

### 3. **Preprocessing متفاوت**
- **مشکل**: ممکن است augmentation یا preprocessing متفاوتی استفاده شده باشد
- **راه‌حل**: در این fine-tuning از preprocessing ساده استفاده می‌شود

### 4. **MMPose Dependency**
- **مشکل**: مدل CLdetection2023 نیاز به MMPose دارد
- **راه‌حل**: اگر MMPose نباشد، از ResNet18 pretrained به عنوان fallback استفاده می‌شود

## 🚀 نحوه استفاده

### روش 1: استفاده از فایل Batch (ساده)

```batch
finetune_p1_p2_cldetection.bat
```

### روش 2: استفاده مستقیم از Python

```bash
python finetune_p1_p2_cldetection.py ^
    --cldetection-model "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\CLdetection2023\model_pretrained_on_train_and_val.pth" ^
    --annotations annotations_p1_p2.json ^
    --image-dir Aariz/train/Cephalograms ^
    --batch-size 16 ^
    --epochs 100
```

## ⚙️ پارامترهای قابل تنظیم

| پارامتر | پیش‌فرض | توضیح |
|---------|---------|-------|
| `--cldetection-model` | مسیر مدل CLdetection2023 | مسیر فایل `.pth` مدل |
| `--annotations` | `annotations_p1_p2.json` | فایل JSON annotationها |
| `--image-dir` | `Aariz/train/Cephalograms` | پوشه تصاویر |
| `--image-size` | `1024` | اندازه تصویر (CLdetection2023 default) |
| `--batch-size` | `4` | اندازه batch (کاهش یافته برای 1024x1024) |
| `--epochs` | `100` | تعداد epochها |
| `--lr` | `0.001` | Learning rate |
| `--unfreeze-after` | `None` | بعد از چند epoch backbone را unfreeze کند |

## 🔧 استراتژی Training

### مرحله 1: Training فقط Head (پیش‌فرض)
- Backbone **frozen** است (weights تغییر نمی‌کند)
- فقط head جدید برای p1/p2 آموزش می‌بیند
- سریع‌تر و نیاز به memory کمتر

### مرحله 2: Fine-tuning کامل (اختیاری)
- بعد از چند epoch می‌توانید backbone را unfreeze کنید
- از `--unfreeze-after N` استفاده کنید
- Learning rate برای backbone کمتر است (0.1 × LR اصلی)

## 📊 خروجی

پس از training، فایل `checkpoint_p1_p2_cldetection.pth` ایجاد می‌شود که شامل:
- State dict مدل کامل (backbone + head)
- Optimizer state
- Loss values
- Metadata

## 🔍 Troubleshooting

### مشکل 1: MMPose not found
```
⚠ MMPose not available. Will use alternative approach.
```
**راه‌حل**: مشکلی نیست! از ResNet18 pretrained استفاده می‌شود که هم خوب کار می‌کند.

### مشکل 2: CUDA out of memory
**راه‌حل**: 
- `--batch-size` را کاهش دهید (مثلاً 2 یا 1 برای 1024x1024)
- یا از `--freeze-backbone` استفاده کنید (که پیش‌فرض است)
- توجه: با سایز 1024x1024، batch size پیش‌فرض 4 است

### مشکل 3: Loss کاهش نمی‌یابد
**راه‌حل**:
- Learning rate را کاهش دهید
- بعد از چند epoch backbone را unfreeze کنید
- تعداد epochها را افزایش دهید

### مشکل 4: دقت پایین
**راه‌حل**:
- بعد از چند epoch backbone را unfreeze کنید
- Learning rate را تنظیم کنید
- تعداد داده‌های training را افزایش دهید

## 📈 مقایسه با آموزش از صفر

| روش | مزایا | معایب |
|-----|-------|-------|
| **Fine-tuning CLdetection2023** | ✅ دقت بالاتر<br>✅ سریع‌تر<br>✅ نیاز به داده کمتر | ⚠️ نیاز به MMPose (یا fallback) |
| **آموزش از صفر** | ✅ ساده‌تر<br>✅ مستقل | ❌ نیاز به داده بیشتر<br>❌ زمان بیشتر |

## 💡 توصیه‌ها

1. **ابتدا با backbone frozen شروع کنید** (پیش‌فرض)
2. **اگر دقت کافی نبود**، بعد از 20-30 epoch backbone را unfreeze کنید
3. **Learning rate را تنظیم کنید**: برای head بالاتر، برای backbone پایین‌تر
4. **از validation loss استفاده کنید** برای early stopping

## 🎓 نکات مهم

1. **دیتاست‌های متفاوت**: اگرچه دیتاست‌ها متفاوت هستند، اما:
   - هر دو روی تصاویر cephalometric هستند
   - Features مشترک زیادی دارند
   - Normalization مشترک (ImageNet) استفاده می‌شود

2. **Transfer Learning**: این روش در واقع transfer learning است که:
   - از دانش قبلی مدل استفاده می‌کند
   - فقط head جدید را یاد می‌گیرد
   - معمولاً بهتر از آموزش از صفر عمل می‌کند

3. **Fallback Strategy**: اگر MMPose نباشد:
   - از ResNet18 pretrained استفاده می‌شود
   - هنوز هم transfer learning است
   - نتایج خوبی می‌دهد

## 📝 مثال استفاده

```python
# Load trained model
checkpoint = torch.load('checkpoint_p1_p2_cldetection.pth')
model = P1P2ModelWithCLDetectionBackbone(
    cldetection_model_path='path/to/cldetection.pth',
    device='cuda'
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(image_tensor)  # Shape: [batch, 4]
    # output[:, 0:2] = p1 (x, y)
    # output[:, 2:4] = p2 (x, y)
```

---

**نکته نهایی**: حتی اگر دیتاست‌ها متفاوت باشند، استفاده از backbone pretrained معمولاً بهتر از آموزش از صفر است. این روش transfer learning است که در deep learning بسیار رایج و موثر است.

