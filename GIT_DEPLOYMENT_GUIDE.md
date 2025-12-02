# 🚀 راهنمای انتقال پروژه به سرور اوبونتو با Git

## ✅ وضعیت فعلی
- ✅ فایل‌ها commit شده‌اند
- ⚠️ Repository آماده برای push است

## 📋 روش‌های انتقال

### روش 1: استفاده از GitHub/GitLab (پیشنهادی - ساده‌تر)

#### مرحله 1: ایجاد Repository روی GitHub/GitLab
1. به GitHub.com یا GitLab.com بروید
2. یک repository جدید ایجاد کنید (مثلاً `dental-ai`)
3. URL repository را کپی کنید (مثلاً: `https://github.com/username/dental-ai.git`)

#### مرحله 2: اضافه کردن Remote و Push
```powershell
# روی کامپیوتر شخصی (ویندوز):
git remote add origin https://github.com/username/dental-ai.git
git branch -M main
git push -u origin main
```

#### مرحله 3: Clone روی سرور اوبونتو
```bash
# روی سرور اوبونتو:
cd /var/www  # یا هر مسیر دلخواه
git clone https://github.com/username/dental-ai.git dental-ai
cd dental-ai
```

---

### روش 2: استفاده از Git روی سرور اوبونتو (بدون GitHub)

#### مرحله 1: ایجاد Bare Repository روی سرور
```bash
# روی سرور اوبونتو:
sudo mkdir -p /opt/git/dental-ai.git
cd /opt/git/dental-ai.git
sudo git init --bare
sudo chown -R $USER:$USER /opt/git/dental-ai.git
```

#### مرحله 2: اضافه کردن Remote و Push از کامپیوتر شخصی
```powershell
# روی کامپیوتر شخصی (ویندوز):
# جایگزین کنید: USER@SERVER_IP با اطلاعات سرور شما
git remote add origin ssh://USER@SERVER_IP/opt/git/dental-ai.git
git branch -M main
git push -u origin main
```

#### مرحله 3: Clone روی سرور
```bash
# روی سرور اوبونتو:
cd /var/www  # یا هر مسیر دلخواه
git clone /opt/git/dental-ai.git dental-ai
cd dental-ai
```

---

## 🔧 تنظیمات بعد از Clone

### 1. نصب Dependencies
```bash
# Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_minimal.txt

# Node.js dependencies (Frontend)
cd vite-js
npm install
npm run build

# Node.js dependencies (Backend - اگر وجود دارد)
cd ../minimal-api-dev-v6
npm install
npm run build
```

### 2. تنظیم Environment Variables
```bash
# کپی کردن فایل env.example
cp env.example .env
# ویرایش .env با اطلاعات سرور
nano .env
```

### 3. اجرای Deployment Script
```bash
# اگر اسکریپت deployment دارید:
chmod +x deploy-ubuntu.sh
./deploy-ubuntu.sh
```

---

## 🔄 به‌روزرسانی پروژه در آینده

### از کامپیوتر شخصی:
```powershell
git add .
git commit -m "Update message"
git push origin main
```

### روی سرور اوبونتو:
```bash
cd /var/www/dental-ai  # یا مسیر پروژه
git pull origin main
# سپس dependencies را به‌روز کنید اگر نیاز باشد
```

---

## ⚠️ نکات مهم

1. **فایل‌های حساس**: فایل‌های `.env` در `.gitignore` هستند و به Git اضافه نمی‌شوند
2. **فایل‌های بزرگ**: فایل‌های مدل (`.pt`, `.onnx`) در `.gitignore` هستند
3. **SSH Key**: برای استفاده از SSH، باید SSH key را روی سرور تنظیم کنید
4. **Permissions**: مطمئن شوید که کاربر روی سرور دسترسی لازم را دارد

---

## 🆘 عیب‌یابی

### خطای "Permission denied"
```bash
# روی سرور:
sudo chown -R $USER:$USER /opt/git/dental-ai.git
```

### خطای "Repository not found"
- مطمئن شوید URL repository درست است
- برای GitHub/GitLab، مطمئن شوید repository public است یا SSH key تنظیم شده

### خطای "Corrupt loose object"
```powershell
# روی کامپیوتر شخصی:
git fsck --full
git gc --aggressive --prune=now
```

---

## 📞 اطلاعات مورد نیاز

برای ادامه، لطفاً مشخص کنید:
1. **آیا می‌خواهید از GitHub/GitLab استفاده کنید؟** (اگر بله، URL repository را بدهید)
2. **یا می‌خواهید repository را روی سرور خودتان راه‌اندازی کنید؟** (اگر بله، IP سرور و username را بدهید)


