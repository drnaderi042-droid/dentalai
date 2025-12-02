# راهنمای تست کامل HRNet - مقایسه Python Direct vs Frontend API vs Ground Truth

این اسکریپت مدل HRNet را به **سه حالت مختلف** تست می‌کند و نتایج را با Ground Truth مقایسه می‌کند:

## 🎯 حالت‌های تست

### 1. **python** - فقط تست Python Direct
- تست مستقیم مدل از طریق Python (بدون API)
- مقایسه نتایج با Ground Truth
- مناسب برای تست سریع و بدون نیاز به API Server

### 2. **frontend** - فقط تست Frontend API
- تست مدل از طریق API (شبیه‌سازی فرانت‌اند)
- مقایسه نتایج با Ground Truth
- نیاز به اجرای API Server

### 3. **all** - تست هر دو و مقایسه کامل
- تست Python Direct
- تست Frontend API
- مقایسه هر دو با Ground Truth
- مقایسه دو روش با یکدیگر
- **این حالت پیش‌فرض است**

## 📋 پیش‌نیازها

### 1. نصب وابستگی‌ها

```bash
cd cephx_service
python -m venv venv
venv\Scripts\activate  # Windows
# یا
source venv/bin/activate  # Linux/Mac

pip install torch torchvision
pip install -r requirements_hrnet.txt
```

### 2. راه‌اندازی API Server

برای تست Frontend API، باید سرور API را اجرا کنید:

```bash
cd cephx_service
python app_hrnet_real.py
```

سرور روی `http://localhost:5000` اجرا می‌شود.

## 🚀 اجرای تست

### روش 1: استفاده از فایل Batch (Windows)

```bash
cd Aariz
run_python_frontend_comparison_test.bat
```

سپس حالت مورد نظر را انتخاب کنید:
- `1` برای Python Direct
- `2` برای Frontend API
- `3` برای هر دو (پیش‌فرض)

### روش 2: اجرای مستقیم Python

```bash
cd Aariz

# تست فقط Python Direct
python test_hrnet_python_frontend_comparison.py --mode python

# تست فقط Frontend API
python test_hrnet_python_frontend_comparison.py --mode frontend

# تست هر دو و مقایسه (پیش‌فرض)
python test_hrnet_python_frontend_comparison.py --mode all

# یا بدون مشخص کردن mode (پیش‌فرض: all)
python test_hrnet_python_frontend_comparison.py
```

### روش 3: با Virtual Environment

```bash
cd cephx_service
venv\Scripts\activate
cd ..\Aariz
python test_hrnet_python_frontend_comparison.py --mode python
```

## ⚙️ تنظیمات

می‌توانید تصویر تست را تغییر دهید:

```bash
# تست با تصویر دیگر
python test_hrnet_python_frontend_comparison.py --mode all --image-id YOUR_IMAGE_ID
```

یا در ابتدای فایل `test_hrnet_python_frontend_comparison.py`:

```python
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"  # تغییر دهید
```

## 📊 خروجی

اسکریپت نتایج زیر را نمایش می‌دهد:

### 1. نتایج Python Direct
- جدول مقایسه با Ground Truth
- آمار خطاها (MRE, Median, Min, Max, Std Dev)
- Success Detection Rate (SDR) برای آستانه‌های مختلف

### 2. نتایج Frontend API
- جدول مقایسه با Ground Truth
- آمار خطاها
- Success Detection Rate

### 3. مقایسه دو روش
- مقایسه MRE، Median، و سایر معیارها
- مقایسه SDR برای آستانه‌های مختلف
- تعیین روش بهتر

### 4. فایل JSON خروجی

نتایج کامل در فایل `hrnet_test_results_{mode}_{image_id}.json` ذخیره می‌شود که شامل:
- Ground Truth landmarks
- نتایج Python Direct با خطاها (اگر تست شده باشد)
- نتایج Frontend API با خطاها (اگر تست شده باشد)
- آمار کامل برای هر روش
- حالت تست استفاده شده

## 📈 معیارهای ارزیابی

### MRE (Mean Radial Error)
میانگین فاصله اقلیدسی بین لندمارک‌های پیش‌بینی شده و Ground Truth بر حسب میلی‌متر.

### SDR (Success Detection Rate)
درصد لندمارک‌هایی که خطای آن‌ها کمتر از آستانه مشخص است:
- SDR @ 1mm: لندمارک‌هایی با خطا ≤ 1mm
- SDR @ 2mm: لندمارک‌هایی با خطا ≤ 2mm
- SDR @ 2.5mm: لندمارک‌هایی با خطا ≤ 2.5mm
- SDR @ 3mm: لندمارک‌هایی با خطا ≤ 3mm
- SDR @ 4mm: لندمارک‌هایی با خطا ≤ 4mm

## 🔍 Mapping لندمارک‌ها

اسکریپت به صورت خودکار لندمارک‌های HRNet را به لندمارک‌های Ground Truth نگاشت می‌کند:

| HRNet | Ground Truth | توضیحات |
|-------|--------------|---------|
| S | S | Sella |
| N | N | Nasion |
| Or | Or | Orbitale |
| Po | Po | Porion |
| A | A | A-point |
| B | B | B-point |
| Pog | Pog | Pogonion |
| Me | Me | Menton |
| Gn | Gn | Gnathion |
| Go | Go | Gonion |
| L1 | LIT | Lower Incisor Tip |
| U1 | UIT | Upper Incisor Tip |
| UL | Ls | Upper Lip (Labrale superius) |
| LL | Li | Lower Lip (Labrale inferius) |
| Sn | Sn | Subnasale |
| PogSoft | Pog` | Soft Tissue Pogonion |
| PNS | PNS | Posterior Nasal Spine |
| ANS | ANS | Anterior Nasal Spine |
| Ar | Ar | Articulare |

## ⚠️ مشکلات رایج

### 1. خطای Connection Error
```
❌ Connection error! API service may not be running
```

**راه حل**: سرور API را اجرا کنید:
```bash
cd cephx_service
python app_hrnet_real.py
```

### 2. خطای Import Error
```
❌ Import error: No module named 'hrnet_production_service'
```

**راه حل**: مطمئن شوید که از دایرکتوری صحیح اجرا می‌کنید و virtual environment فعال است.

### 3. خطای Model Not Found
```
❌ ERROR: Model file not found
```

**راه حل**: مطمئن شوید که فایل `hrnet_cephalometric.pth` در `cephx_service/model/` وجود دارد.

### 4. خطای Ground Truth Not Found
```
❌ ERROR: Ground Truth not found
```

**راه حل**: مطمئن شوید که:
- تصویر تست در `Aariz/Aariz/train/Cephalograms/` وجود دارد
- فایل Ground Truth در `Aariz/Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists/` وجود دارد

## 📝 مثال خروجی

```
================================================================================
📊 مقایسه نتایج Python Direct با Ground Truth
================================================================================

Landmark     Pred X       Pred Y       GT X       GT Y       Diff X     Diff Y     Error (px)  Error (mm)  Conf    
----------------------------------------------------------------------------------------------------
S            499.23       758.45       499         758        0.23       0.45       0.50        0.0500      0.923
N            1183.12     508.23       1183        508        0.12       0.23       0.26        0.0260      0.891
...

================================================================================
📈 آمار خطاها (Python Direct)
================================================================================

✅ تعداد لندمارک‌های مقایسه شده: 19

📊 خطا بر حسب میلی‌متر:
   میانگین (MRE): 0.6234 mm
   میانه: 0.5123 mm
   کمینه: 0.0123 mm
   بیشینه: 2.3456 mm
   انحراف معیار: 0.4567 mm

================================================================================
📊 Success Detection Rate (SDR) - Python Direct
================================================================================
   SDR @ 1.0mm: 89.47% (17/19)
   SDR @ 2.0mm: 94.74% (18/19)
   SDR @ 2.5mm: 100.00% (19/19)
   ...
```

## 🎓 استفاده برای توسعه

این اسکریپت می‌تواند برای:
- تست تغییرات در مدل
- مقایسه نسخه‌های مختلف مدل
- بررسی تأثیر پیش‌پردازش‌ها
- ارزیابی عملکرد API
- دیباگ مشکلات دقت

استفاده شود.

## 📞 پشتیبانی

در صورت بروز مشکل، لطفاً:
1. لاگ‌های کامل را بررسی کنید
2. مطمئن شوید که تمام پیش‌نیازها نصب شده‌اند
3. بررسی کنید که سرور API در حال اجرا است (برای تست Frontend API)

