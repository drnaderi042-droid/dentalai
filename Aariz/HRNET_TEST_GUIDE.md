# راهنمای تست HRNet

## 📋 اسکریپت‌های تست

### 1. `test_hrnet_direct.py` ⭐ **توصیه می‌شود**
**تست مستقیم Real Model (بدون API)**
- استفاده مستقیم از `HRNetProductionService`
- بدون نیاز به Flask API
- مطمئن‌ترین روش برای تست

### 2. `test_hrnet_complete_comparison.py`
**مقایسه API vs Direct**
- تست هم از طریق API
- تست هم مستقیم
- مقایسه نتایج

### 3. `test_hrnet_full_comparison.py`
**تست از طریق API فقط**
- فقط از Flask API استفاده می‌کند
- اگر API mock باشد، نتایج نادرست می‌دهد

---

## 🚀 نحوه اجرا

### روش 1: استفاده از Batch Files ⭐ **توصیه می‌شود**

Batch files به صورت خودکار از **venv** استفاده می‌کنند (مشکل `easydict` حل شده)

```batch
# دوبار کلیک کنید یا در PowerShell اجرا کنید:
Aariz\run_hrnet_direct_test.bat          ⭐ تست مستقیم (Real Model)
Aariz\run_hrnet_complete_test.bat        تست کامل (API + Direct)
Aariz\run_hrnet_test.bat                 تست API
```

**✅ این batch files:**
- به صورت خودکار از `cephx_service\venv` استفاده می‌کنند
- `easydict` و سایر dependencies را پیدا می‌کنند
- نیاز به activate کردن venv نیست!

### روش 2: اجرای مستقیم در PowerShell (با venv)

```powershell
# از دایرکتوری اصلی پروژه:
cd "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy"

# استفاده از Python از venv:
cephx_service\venv\Scripts\python.exe Aariz\test_hrnet_direct.py
cephx_service\venv\Scripts\python.exe Aariz\test_hrnet_complete_comparison.py
cephx_service\venv\Scripts\python.exe Aariz\test_hrnet_full_comparison.py
```

**یا فعال کردن venv:**
```powershell
cd cephx_service
.\venv\Scripts\Activate.ps1
cd ..\Aariz
python test_hrnet_direct.py
```

### روش 3: از دایرکتوری اصلی

```powershell
# از هر دایرکتوری
python "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\Aariz\test_hrnet_direct.py"
```

---

## 📊 نتایج انتظاری

### ✅ Real Model (تست مستقیم)
- **Model Type**: `real`
- **Input Size**: `[768, 768]`
- **MRE**: < 2mm
- **SDR @ 2mm**: > 70%

### ❌ Mock Model (اگر از API اشتباه استفاده شود)
- **Model Type**: `mock` یا `N/A`
- **Input Size**: `N/A`
- **MRE**: > 30mm (بسیار بالا!)
- **SDR @ 2mm**: 0%

---

## 🔍 تشخیص Mock vs Real

در خروجی اسکریپت، به دنبال این موارد باشید:

### ✅ Real Model:
```
✅ Using REAL HRNet model
Model Type: real
Input Size: [768, 768]
MRE: 0.63mm (یا عددی نزدیک)
```

### ❌ Mock Model:
```
⚠️  WARNING: Using MOCK model!
Model Type: mock
Input Size: N/A
MRE: 39.08mm (یا عددی بسیار بالا)
```

---

## ⚠️ نکات مهم

1. **برای تست Real Model**: از `test_hrnet_direct.py` استفاده کنید
   - نیازی به Flask API ندارد
   - مستقیماً از `HRNetProductionService` استفاده می‌کند

2. **برای تست API**: مطمئن شوید سرویس Real اجرا می‌شود
   ```batch
   cephx_service\run_hrnet_service.bat
   ```
   این باید `app_hrnet_real.py` را اجرا کند (نه `app_hrnet.py`)

3. **بررسی سرویس**: اگر از API استفاده می‌کنید، اول `/health` را چک کنید:
   ```
   http://localhost:5000/health
   ```
   باید `"model_type": "real"` باشد

---

## 🐛 عیب‌یابی

### مشکل 1: `ModuleNotFoundError: No module named 'easydict'`
**✅ حل شد!** Batch files به صورت خودکار از venv استفاده می‌کنند.

**اگر هنوز مشکل دارید:**
```powershell
# استفاده از batch file (توصیه می‌شود):
Aariz\run_hrnet_direct_test.bat

# یا فعال کردن venv:
cd cephx_service
.\venv\Scripts\Activate.ps1
cd ..\Aariz
python test_hrnet_direct.py
```

### مشکل 2: `Torch not compiled with CUDA enabled`
**✅ حل شد!** اسکریپت به صورت خودکار از CPU استفاده می‌کند اگر CUDA در دسترس نباشد.

**نکته:** استفاده از CPU کندتر است اما کار می‌کند. برای سرعت بیشتر، PyTorch با CUDA نصب کنید.

### مشکل 3: `FileNotFoundError: Model checkpoint`
```powershell
# بررسی کنید checkpoint وجود دارد:
Test-Path "cephx_service\model\hrnet_cephalometric.pth"
```

### مشکل 4: نتایج Mock Model
- سرویس Flask را بررسی کنید
- مطمئن شوید `app_hrnet_real.py` اجرا می‌شود
- یا از `test_hrnet_direct.py` استفاده کنید (نیازی به API ندارد)

---

## 📝 نمونه خروجی موفق

```
================================================================================
🧪 تست مستقیم HRNet Model (Real Model - بدون API)
================================================================================

📸 تصویر تست: cks2ip8fq29yq0yufc4scftj8

🤖 بارگذاری HRNet Model...
   ✅ Model loaded successfully!
   Model Type: REAL (not mock)
   Input Size: (768, 768)
   Accuracy (from checkpoint): 0.6300mm

🔍 اجرای تشخیص...
   ✅ Detection complete!
   Valid landmarks: 19/19

📊 نتایج مقایسه:
   MRE: 0.8543 mm
   SDR @ 2mm: 89.47%

✅ نتایج عالی! MRE کمتر از 2mm است
```

---

**تاریخ**: 2024-11-01

