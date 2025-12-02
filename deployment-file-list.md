# 📋 لیست فایل‌های مورد نیاز برای Deployment DentalAI

## 🎯 پروژه شما شامل ۳ کامپوننت اصلی است:

### 1. **Frontend (vite-js)** - پورت 3030
### 2. **Backend API (minimal-api-dev-v6)** - پورت 7272
### 3. **AI Server (unified_ai_api_server.py)** - پورت 5000

---

## 📦 فایل‌های ضروری برای انتقال به سرور:

### **Frontend - فایل‌های Build شده:**
```
📁 vite-js/dist/
├── assets/           # فایل‌های JavaScript و CSS کامپایل شده
├── fonts/           # فونت‌ها
├── logo/            # لوگوها
├── payment-icons/   # آیکون‌های پرداخت
├── teeth/           # تصاویر دندان‌ها
├── favicon.ico      # آیکون سایت
└── index.html       # فایل اصلی HTML
```

### **Backend API - فایل‌های Build شده:**
```
📁 minimal-api-dev-v6/.next/
├── server/          # فایل‌های سرور
├── static/          # فایل‌های static
├── cache/           # کش build
├── BUILD_ID         # شناسه build
├── build-manifest.json
├── routes-manifest.json
└── [فایل‌های دیگر build]
```

### **Backend API - فایل‌های تنظیمات:**
```
📁 minimal-api-dev-v6/
├── package.json     # وابستگی‌ها
├── next.config.mjs  # تنظیمات Next.js
├── prisma/          # تنظیمات دیتابیس
│   ├── schema.prisma
│   └── migrations/
├── src/             # کد منبع (برای development)
└── public/          # فایل‌های static
```

### **AI Server - فایل‌های پایتون:**
```
📁 Root Project/
├── unified_ai_api_server.py          # فایل اصلی سرور AI
├── requirements_unified_api.txt       # وابستگی‌های پایتون
├── cephx_service/                     # سرویس تحلیل سفالومتری
├── facial-landmark-detection/         # تشخیص نقاط چهره
├── CLdetection2023/                   # مدل تشخیص دندان
└── models/                            # مدل‌های AI (اگر وجود دارد)
```

### **فایل‌های تنظیمات عمومی:**
```
📁 Root Project/
├── env.example                        # نمونه متغیرهای محیطی
├── docker-compose.yml                 # تنظیمات Docker
├── Dockerfile.python                  # Docker برای AI server
├── Dockerfile.api                     # Docker برای Backend
├── Dockerfile.frontend                # Docker برای Frontend
├── nginx/nginx.conf                   # تنظیمات Nginx
└── quick-start.sh                     # اسکریپت راه‌اندازی سریع
```

---

## 🎯 استراتژی انتقال فایل‌ها:

### **روش ۱: انتقال انتخابی (توصیه شده):**

#### **مرحله ۱: ایجاد ساختار دایرکتوری روی سرور:**
```bash
# روی سرور Ubuntu
mkdir -p ~/dentalai/{frontend,backend,ai-server,nginx}
```

#### **مرحله ۲: انتقال Frontend:**
```bash
# از ویندوز - انتقال فایل‌های build شده
scp -r "C:\path\to\vite-js\dist" user@server:~/dentalai/frontend/
```

#### **مرحله ۳: انتقال Backend API:**
```bash
# انتقال فایل‌های build شده
scp -r "C:\path\to\minimal-api-dev-v6\.next" user@server:~/dentalai/backend/
scp "C:\path\to\minimal-api-dev-v6\package.json" user@server:~/dentalai/backend/
scp "C:\path\to\minimal-api-dev-v6\next.config.mjs" user@server:~/dentalai/backend/
scp -r "C:\path\to\minimal-api-dev-v6\prisma" user@server:~/dentalai/backend/
```

#### **مرحله ۴: انتقال AI Server:**
```bash
# انتقال فایل‌های پایتون
scp "C:\path\to\unified_ai_api_server.py" user@server:~/dentalai/ai-server/
scp "C:\path\to\requirements_unified_api.txt" user@server:~/dentalai/ai-server/
scp -r "C:\path\to\cephx_service" user@server:~/dentalai/ai-server/
scp -r "C:\path\to\facial-landmark-detection" user@server:~/dentalai/ai-server/
scp -r "C:\path\to\CLdetection2023" user@server:~/dentalai/ai-server/
```

#### **مرحله ۵: انتقال فایل‌های تنظیمات:**
```bash
# انتقال فایل‌های تنظیمات
scp "C:\path\to\env.example" user@server:~/dentalai/
scp "C:\path\to\docker-compose.yml" user@server:~/dentalai/
scp -r "C:\path\to\nginx" user@server:~/dentalai/
```

### **روش ۲: انتقال کامل (ساده‌تر):**

```bash
# انتقال کل پروژه (بزرگ‌تر اما ساده‌تر)
scp -r "C:\path\to\project" user@server:~/dentalai-project/

# روی سرور - انتقال فایل‌های build شده به مکان نهایی
cp -r ~/dentalai-project/vite-js/dist ~/dentalai/frontend/
cp -r ~/dentalai-project/minimal-api-dev-v6/.next ~/dentalai/backend/
cp ~/dentalai-project/unified_ai_api_server.py ~/dentalai/ai-server/
# ... سایر فایل‌ها
```

---

## 📂 ساختار نهایی روی سرور:

```
/home/user/
├── dentalai/
│   ├── frontend/
│   │   └── dist/          # فایل‌های build شده Vite
│   ├── backend/
│   │   ├── .next/         # فایل‌های build شده Next.js
│   │   ├── package.json
│   │   ├── next.config.mjs
│   │   └── prisma/
│   ├── ai-server/
│   │   ├── unified_ai_api_server.py
│   │   ├── requirements_unified_api.txt
│   │   ├── cephx_service/
│   │   ├── facial-landmark-detection/
│   │   └── CLdetection2023/
│   ├── nginx/
│   │   └── nginx.conf
│   ├── docker-compose.yml
│   └── env.example
```

---

## 📋 چک‌لیست انتقال:

### **Frontend:**
- [ ] `vite-js/dist/` انتقال داده شده
- [ ] شامل `index.html`, `assets/`, `fonts/` و غیره

### **Backend API:**
- [ ] `minimal-api-dev-v6/.next/` انتقال داده شده
- [ ] `package.json`, `next.config.mjs` انتقال داده شده
- [ ] `prisma/` انتقال داده شده

### **AI Server:**
- [ ] `unified_ai_api_server.py` انتقال داده شده
- [ ] `requirements_unified_api.txt` انتقال داده شده
- [ ] فولدرهای `cephx_service/`, `facial-landmark-detection/`, `CLdetection2023/` انتقال داده شده

### **تنظیمات:**
- [ ] `env.example` انتقال داده شده و به `.env` تغییر نام داده شده
- [ ] فایل‌های Docker انتقال داده شده
- [ ] تنظیمات Nginx انتقال داده شده

---

## 💾 حجم تقریبی فایل‌ها:

- **Frontend (dist/)**: ~۵۰-۱۰۰MB
- **Backend (.next/)**: ~۱۰۰-۲۰۰MB
- **AI Server**: ~۵۰۰MB-۲GB (بسته به مدل‌ها)
- **کل پروژه**: ~۵۰۰MB-۲GB

---

## 🚀 بعد از انتقال:

### **روی سرور Ubuntu:**

```bash
# ۱. رفتن به دایرکتوری پروژه
cd ~/dentalai

# ۲. تنظیم متغیرهای محیطی
cp env.example .env
nano .env  # تنظیم مقادیر

# ۳. راه‌اندازی با Docker
docker-compose up -d

# یا راه‌اندازی دستی
./setup-and-deploy.sh
```

---

## ⚡ نکات مهم:

1. **ترتیب انتقال**: فایل‌های build شده را اول انتقال دهید
2. **فشرده‌سازی**: برای انتقال سریع از `tar` استفاده کنید:
   ```bash
   # فشرده‌سازی روی ویندوز
   tar -czf dentalai-build.tar.gz dist/ .next/ unified_ai_api_server.py
   
   # انتقال
   scp dentalai-build.tar.gz user@server:~/
   
   # استخراج روی سرور
   tar -xzf dentalai-build.tar.gz
   ```

3. **امنیت**: فایل‌های `.env` را با دقت انتقال دهید
4. **Backup**: قبل از انتقال، از فایل‌های build شده backup بگیرید

حالا آماده انتقال فایل‌ها به سرور هستید! 🎯



