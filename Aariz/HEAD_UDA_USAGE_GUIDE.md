# راهنمای تست مدل head_uda.pth

## مشکل

مدل `head_uda.pth` از repository [UDA_Med_Landmark](https://github.com/jhb86253817/UDA_Med_Landmark) دارای ساختار متفاوتی است:
- **معماری**: ResNet + Transformer Decoder + UDA (Unsupervised Domain Adaptation)
- **تعداد لندمارک‌ها**: 19 لندمارک
- **خروجی**: `cls_layer`, `x_layer`, `y_layer` برای پیش‌بینی کلاس و مختصات

این مدل نمی‌تواند مستقیماً با کد فعلی شما استفاده شود.

## راه حل

### گزینه 1: استفاده از Repository اصلی (پیشنهادی)

1. **Clone کردن repository اصلی**:
```bash
git clone https://github.com/jhb86253817/UDA_Med_Landmark.git
cd UDA_Med_Landmark
```

2. **نصب dependencies**:
```bash
conda create -n uda_med_landmark python=3.9
conda activate uda_med_landmark
pip install -r requirements.txt
```

3. **کپی کردن مدل**:
```bash
cp /path/to/head_uda.pth UDA_Med_Landmark/weights/
```

4. **تست مدل**:
از اسکریپت‌های موجود در repository استفاده کنید.

### گزینه 2: استفاده از مدل‌های موجود شما

به جای استفاده از `head_uda.pth`، می‌توانید از مدل‌های آموزش‌داده‌شده خودتان استفاده کنید:
- `checkpoints/checkpoint_best.pth` (29 لندمارک)
- `checkpoints_512x512/checkpoint_best.pth`
- `checkpoints_768x768/checkpoint_best.pth`

این مدل‌ها با دیتاست Aariz شما سازگار هستند و 29 لندمارک را تشخیص می‌دهند.

## مقایسه

| ویژگی | head_uda.pth | مدل‌های شما |
|-------|--------------|-------------|
| تعداد لندمارک | 19 | 29 |
| معماری | ResNet + Transformer | HRNet/ResNet |
| سازگاری با Aariz | نیاز به mapping | ✅ مستقیم |
| دقت روی Aariz | نامشخص | ✅ تست شده |

## توصیه

**استفاده از مدل‌های خودتان** بهتر است چون:
1. ✅ با دیتاست Aariz سازگار هستند
2. ✅ 29 لندمارک را تشخیص می‌دهند (نه 19)
3. ✅ قبلاً تست شده‌اند
4. ✅ نیازی به repository خارجی ندارند

## اگر می‌خواهید head_uda.pth را تست کنید

می‌توانید از repository اصلی استفاده کنید یا اگر می‌خواهید، می‌توانم یک wrapper برای استفاده از این مدل بنویسم (اما نیاز به پیاده‌سازی معماری کامل دارد).
















