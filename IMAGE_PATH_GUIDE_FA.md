# راهنمای مسیر تصاویر 📁

## ✅ مسیر صحیح

بله، مسیر شما **درست** است:

```
C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\my_images
```

## 📋 ساختار مورد نیاز

پوشه `my_images` باید شامل این فایل‌ها باشد:

```
my_images/
├── 0.png  # Anterior view (جلو)
├── 1.png  # Left buccal view (چپ)
├── 2.png  # Right buccal view (راست)
├── 3.png  # Maxillary occlusal view (فوقانی)
└── 4.png  # Mandibular occlusal view (تحتانی)
```

## 🚀 استفاده در دستور

### از پوشه TeethDreamer:

```bash
cd TeethDreamer
python seg_teeth.py --img ../my_images --seg ../output/segmented --suffix png
```

**توضیح:**
- `../my_images` = یک پوشه بالاتر، سپس `my_images`
- از `TeethDreamer/` به `main - Copy/my_images/` می‌رود ✅

### از root directory:

```bash
# اگر در root هستید
python TeethDreamer\seg_teeth.py --img my_images --seg output\segmented --suffix png
```

## 🔍 بررسی مسیر

```powershell
# بررسی وجود پوشه
Test-Path "my_images"

# لیست فایل‌ها
Get-ChildItem -Path "my_images" -Filter "*.png" | Select-Object Name

# بررسی فایل‌های مورد نیاز
$required = @("0.png", "1.png", "2.png", "3.png", "4.png")
foreach ($file in $required) {
    $exists = Test-Path "my_images\$file"
    Write-Host "$file : $(if($exists){'✅ موجود'}{'❌ موجود نیست'})"
}
```

## ⚠️ نکات مهم

1. **نام فایل‌ها:** باید دقیقاً `0.png`, `1.png`, `2.png`, `3.png`, `4.png` باشند
2. **فرمت:** باید PNG یا JPG باشند (بسته به `--suffix`)
3. **مسیر نسبی:** از `TeethDreamer/` استفاده کنید: `../my_images`

## ✅ دستور نهایی

```bash
# فعال‌سازی محیط
venv_teethdreamer\Scripts\activate

# ورود به پوشه TeethDreamer
cd TeethDreamer

# اجرای segmentation
python seg_teeth.py --img ../my_images --seg ../output/segmented --suffix png
```

---

**مسیر شما درست است! ✅**













