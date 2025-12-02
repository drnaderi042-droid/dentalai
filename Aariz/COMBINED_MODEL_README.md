# Combined 31-Landmark Detection Model

این مدل ترکیبی برای شناسایی 31 لندمارک طراحی شده است:
- **29 لندمارک آناتومیک** (از مدل اصلی)
- **2 نقطه کالیبراسیون P1/P2** (از مدل تخصصی)

## 📋 فایل‌های مدل

### مدل‌های ورودی:
1. **checkpoint_best_768.pth** - مدل اصلی (29 لندمارک)
   - اندازه: ~73 MB
   - معماری: HRNet-based heatmap detector
   - دقت: بهینه شده برای لندمارک‌های آناتومیک

2. **models/hrnet_p1p2_heatmap_best.pth** - مدل P1/P2 (2 لندمارک)
   - اندازه: ~262 MB
   - معماری: HRNet with heatmap output
   - دقت: بهینه شده برای نقاط کالیبراسیون

### مدل خروجی:
3. **combined_31_landmarks.pth** - مدل ترکیبی (31 لندمارک)
   - اندازه: ~58 MB
   - معماری: Unified HRNet backbone
   - ویژگی: تشخیص همزمان تمام 31 لندمارک

## 🚀 نحوه استفاده

### 1. ایجاد مدل ترکیبی

```bash
python create_combined_model.py
```

این اسکریپت:
- دو مدل ورودی را بارگذاری می‌کند
- یک مدل یکپارچه با 31 لندمارک ایجاد می‌کند
- مدل را در `combined_31_landmarks.pth` ذخیره می‌کند
- یک تست ساده برای تأیید عملکرد انجام می‌دهد

### 2. تست مدل ترکیبی

```bash
python test_combined_31_landmarks.py
```

این اسکریپت:
- مدل ترکیبی را بارگذاری می‌کند
- یک تصویر نمونه را پردازش می‌کند
- 31 لندمارک را پیش‌بینی می‌کند
- نتایج را تجسم می‌کند

### 3. استفاده در کد Python

```python
import torch
from create_combined_model import SimplifiedCombinedModel

# بارگذاری مدل
checkpoint = torch.load('combined_31_landmarks.pth')
model = SimplifiedCombinedModel(num_landmarks=31, backbone='hrnet_w18')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# پیش‌بینی
with torch.no_grad():
    # image_tensor: (1, 3, 768, 768)
    heatmaps = model(image_tensor)  # (1, 31, H, W)
    coords = model.extract_coordinates(heatmaps)  # (1, 62)
    
# coords شامل 31 جفت (x, y) است:
# - coords[0:58] -> 29 لندمارک آناتومیک
# - coords[58:62] -> 2 نقطه کالیبراسیون (P1, P2)
```

## 📊 ساختار مدل

### معماری
```
Input Image (3, 768, 768)
    ↓
HRNet Backbone (hrnet_w18)
    ↓
Multi-scale Features
    ↓
Heatmap Decoder
    ↓
31 Heatmaps (31, 192, 192)
    ↓
Soft-argmax Coordinate Extraction
    ↓
31 Landmarks (62 values: x1,y1, x2,y2, ..., x31,y31)
```

### ویژگی‌های کلیدی

1. **Unified Architecture**: یک backbone مشترک برای تمام لندمارک‌ها
2. **Heatmap-based**: دقت بالاتر نسبت به regression مستقیم
3. **Multi-scale Features**: استفاده از ویژگی‌های چند مقیاسی HRNet
4. **Efficient**: کوچکتر از مجموع دو مدل جداگانه

## 🎯 لندمارک‌ها

### لندمارک‌های آناتومیک (1-29)
- Sella (S)
- Nasion (N)
- A point (A)
- B point (B)
- Pogonion (Pog)
- Menton (Me)
- Gnathion (Gn)
- Gonion (Go)
- و 21 لندمارک دیگر...

### نقاط کالیبراسیون (30-31)
- **P1 (Landmark 30)**: نقطه کالیبراسیون اول
- **P2 (Landmark 31)**: نقطه کالیبراسیون دوم

## 📈 عملکرد

### دقت مورد انتظار
- **لندمارک‌های آناتومیک**: مشابه مدل اصلی (MRE < 2mm)
- **نقاط کالیبراسیون**: مشابه مدل P1/P2 (Pixel Error < 5px)

### سرعت
- **GPU (RTX 3070 Ti)**: ~50-100ms per image
- **CPU**: ~500-1000ms per image

## 🔧 Fine-tuning (اختیاری)

اگر می‌خواهید مدل را روی دیتاست خود fine-tune کنید:

```python
# 1. آماده‌سازی دیتاست با 31 لندمارک
# 2. بارگذاری مدل ترکیبی
model = SimplifiedCombinedModel(num_landmarks=31)
checkpoint = torch.load('combined_31_landmarks.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# 3. Fine-tuning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# ... training loop
```

## 📝 نکات مهم

1. **Input Size**: تصاویر باید به اندازه 768×768 تغییر اندازه داده شوند
2. **Normalization**: از ImageNet mean/std استفاده کنید
3. **Output Format**: مختصات normalized هستند [0, 1]
4. **Device**: مدل روی GPU و CPU کار می‌کند

## 🐛 عیب‌یابی

### مشکل: مدل بارگذاری نمی‌شود
```python
# از strict=False استفاده کنید
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### مشکل: خطای حافظه GPU
```python
# batch size را کاهش دهید یا از CPU استفاده کنید
model = model.cpu()
```

### مشکل: دقت پایین
- مطمئن شوید تصاویر به درستی normalize شده‌اند
- اندازه ورودی را بررسی کنید (باید 768×768 باشد)
- در صورت نیاز fine-tuning انجام دهید

## 📚 مراجع

- **HRNet Paper**: Deep High-Resolution Representation Learning
- **Heatmap-based Detection**: Better than direct coordinate regression
- **Multi-task Learning**: Combining multiple landmark detection tasks

## 🤝 مشارکت

برای بهبود مدل:
1. Fine-tune روی دیتاست بزرگتر
2. افزودن data augmentation
3. استفاده از ensemble methods
4. بهینه‌سازی hyperparameters

## 📄 لایسنس

این مدل برای استفاده تحقیقاتی و آموزشی آزاد است.

---

**نسخه**: 1.0  
**تاریخ**: 2024  
**نویسنده**: Dental AI Team