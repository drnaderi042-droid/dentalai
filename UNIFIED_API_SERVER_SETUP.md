# راهنمای راه‌اندازی Unified AI API Server

## ✅ وضعیت فعلی

- ✅ Flask 3.0.3 نصب شده است
- ✅ سرور در حال اجرا روی پورت 5001 است
- ✅ دو instance از سرور در حال اجرا هستند

## 🚀 روش‌های اجرای سرور

### روش 1: استفاده از اسکریپت Batch (توصیه می‌شود)

```bash
run_unified_api_server.bat
```

این اسکریپت به صورت خودکار:
- Flask را بررسی می‌کند
- در صورت نیاز نصب می‌کند
- سرور را اجرا می‌کند

### روش 2: اجرای مستقیم

```bash
python unified_ai_api_server.py
```

### روش 3: با مشخص کردن پورت

```bash
python unified_ai_api_server.py --port 5001
```

## 🔧 حل مشکلات

### مشکل 1: ModuleNotFoundError: No module named 'flask'

**راه حل:**

```bash
# نصب Flask
python -m pip install flask flask-cors

# یا
pip install flask flask-cors
```

### مشکل 2: سرور قبلاً در حال اجرا است

**راه حل:**

```bash
# پیدا کردن process های Python
netstat -ano | findstr :5001

# متوقف کردن process (PID را از خروجی بالا بگیرید)
taskkill /PID <PID> /F
```

### مشکل 3: پورت در حال استفاده است

**راه حل:**

```bash
# استفاده از پورت دیگر
python unified_ai_api_server.py --port 5002
```

## 📋 بررسی نصب Dependencies

برای نصب همه dependencies:

```bash
pip install -r requirements_unified_api.txt
```

Dependencies اصلی:
- flask>=2.0.0,<2.3.0
- flask-cors>=4.0.0
- opencv-python>=4.8.0
- Pillow>=10.0.0
- numpy>=1.24.0
- ultralytics>=8.0.0
- mediapipe>=0.10.0
- torch>=1.9.0
- torchvision>=0.10.0
- scipy>=1.7.0
- scikit-image>=0.18.0

## 🌐 دسترسی به سرور

بعد از اجرای سرور:

- **Local**: http://localhost:5001
- **Network**: http://0.0.0.0:5001
- **Health Check**: http://localhost:5001/health
- **API Docs**: http://localhost:5001/

## 🔍 بررسی وضعیت سرور

```bash
# بررسی پورت
netstat -ano | findstr :5001

# تست health endpoint
curl http://localhost:5001/health
```

## ⚠️ نکات مهم

1. **Virtual Environment**: اگر از virtual environment استفاده می‌کنید، حتماً آن را فعال کنید:
   ```bash
   venv\Scripts\activate
   ```

2. **Python Version**: این پروژه با Python 3.8+ کار می‌کند

3. **Port Conflicts**: اگر پورت 5001 در حال استفاده است، از پورت دیگری استفاده کنید

## 🔧 نصب مدل CLdetection2023

برای استفاده از مدل CLdetection2023، باید MMPose و وابستگی‌های آن را نصب کنید:

### روش 1: استفاده از اسکریپت نصب خودکار (توصیه می‌شود)

```bash
python install_cldetection2023.py
```

### روش 2: نصب دستی

```bash
# 1. نصب openmim
pip install -U openmim

# 2. نصب mmengine
mim install mmengine

# 3. حذف mmcv موجود (در صورت ناسازگار بودن)
pip uninstall mmcv mmcv-full -y

# 4. نصب mmcv (نسخه سازگار)
mim install "mmcv>=2.0.0rc4,<=2.1.0"

# 5. نصب MMPose
cd CLdetection2023/mmpose_package/mmpose
pip install -e .
cd ../../..

# 6. آپگرید numpy
pip install --upgrade numpy

# توجه: SimpleITK اختیاری است و فقط برای لود کردن فایل‌های .mha آموزشی نیاز است
# برای inference، از یک پیاده‌سازی pure numpy استفاده می‌شود
```

### بررسی نصب

پس از نصب، می‌توانید با دستور زیر بررسی کنید:

```python
python -c "import mmengine; import mmpose; print('MMPose installed successfully')"
```

### حل مشکلات CLdetection2023

#### مشکل 1: `No module named 'mmengine'`

**راه حل:**
```bash
pip install -U openmim
mim install mmengine
```

#### مشکل 2: `No module named 'mmpose'`

**راه حل:**
```bash
cd CLdetection2023/mmpose_package/mmpose
pip install -e .
```

#### مشکل 3: `mmcv installation failed` یا `mmcv version incompatible`

**راه حل:**
```bash
# حذف mmcv موجود
pip uninstall mmcv mmcv-full -y

# نصب mmcv با نسخه سازگار
mim install "mmcv>=2.0.0rc4,<=2.1.0"
```

#### مشکل 4: `No module named 'SimpleITK'`

**راه حل:**
SimpleITK اختیاری است و فقط برای لود کردن فایل‌های آموزشی (.mha) نیاز است.
برای inference، از یک پیاده‌سازی pure numpy استفاده می‌شود که به SimpleITK نیاز ندارد.

اگر می‌خواهید SimpleITK را نصب کنید (فقط برای کار با داده‌های آموزشی):
```bash
# استفاده از wheel آماده (توصیه می‌شود)
pip install SimpleITK

# یا استفاده از conda (اگر conda نصب دارید)
conda install -c conda-forge simpleitk
```

**نکته:** در Windows، نصب SimpleITK از source ممکن است به دلیل طولانی بودن مسیرها با خطا مواجه شود. بهتر است از wheel آماده یا conda استفاده کنید.

## 📞 پشتیبانی

در صورت بروز مشکل:
1. بررسی کنید که Python در PATH است: `python --version`
2. بررسی کنید که Flask نصب است: `python -c "import flask"`
3. برای CLdetection2023، بررسی کنید که MMPose نصب است: `python -c "import mmpose"`
4. لاگ‌های خطا را بررسی کنید


