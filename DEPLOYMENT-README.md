# 🚀 راهنمای کامل Deployment پروژه DentalAI

## 📋 نمای کلی پروژه

پروژه DentalAI یک سیستم کامل هوش مصنوعی برای دندانپزشکی و ارتودنسی شامل چندین کامپوننت است:

### 🎯 کامپوننت‌ها
- **Frontend (Vite)**: رابط کاربری اصلی (پورت 3030)
- **Dashboard (Next.js)**: پنل مدیریتی (پورت 3000)
- **API (Next.js + Prisma)**: API بک‌اند با دیتابیس (پورت 3001)
- **AI Server (Python)**: پردازش تصویر و مدل‌های ML (پورت 8000)
- **PostgreSQL**: دیتابیس اصلی

---

## 🐳 روش سریع: Docker Compose

### پیش‌نیازها
- Docker & Docker Compose
- حداقل ۸GB RAM
- حداقل ۵۰GB فضای ذخیره‌سازی

### مراحل راه‌اندازی

```bash
# ۱. کلون کردن پروژه
git clone <repository-url>
cd dentalai-project

# ۲. کپی فایل متغیرهای محیطی
cp env.example .env

# ۳. ویرایش متغیرهای محیطی (اختیاری)
nano .env

# ۴. راه‌اندازی همه سرویس‌ها
docker-compose up -d

# ۵. بررسی وضعیت
docker-compose ps

# ۶. مشاهده لاگ‌ها
docker-compose logs -f
```

### دسترسی به برنامه
- **Frontend**: http://localhost:3030
- **Dashboard**: http://localhost:3000
- **API**: http://localhost:7272
- **AI Server**: http://localhost:5001

---

## 🛠️ روش دستی: راه‌اندازی مرحله‌ای

### مرحله ۱: آماده‌سازی سرور

```bash
# بروزرسانی سیستم
sudo apt update && sudo apt upgrade -y

# نصب Docker (اختیاری)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# نصب Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# نصب Python
sudo apt install python3 python3-pip python3-venv

# نصب PostgreSQL
sudo apt install postgresql postgresql-contrib
```

### مرحله ۲: راه‌اندازی دیتابیس

```bash
# ایجاد کاربر و دیتابیس
sudo -u postgres psql
CREATE DATABASE dentalai;
CREATE USER dentalai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE dentalai TO dentalai_user;
\q

# یا با Docker
docker run -d --name dentalai-postgres \
  -e POSTGRES_DB=dentalai \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:15
```

### مرحله ۳: راه‌اندازی AI Server

```bash
# ایجاد محیط مجازی
python3 -m venv dentalai-env
source dentalai-env/bin/activate

# نصب وابستگی‌ها
pip install -r requirements_unified_api.txt

# راه‌اندازی سرور
python unified_ai_api_server.py
```

### مرحله ۴: راه‌اندازی API

```bash
cd minimal-api-dev-v6

# نصب وابستگی‌ها
npm install

# تنظیم Prisma
npx prisma generate
npx prisma db push

# راه‌اندازی (روی پورت 7272)
npm run dev
```

### مرحله ۵: Build کردن Frontendها

```bash
# Frontend اصلی
cd ../vite-js
npm install && npm run build

# Dashboard
cd ../next-js
npm install && npm run build && npm run start
```

---

## 🔧 تنظیمات پیشرفته

### متغیرهای محیطی

کپی کنید `env.example` به `.env` و مقادیر را تنظیم کنید:

```bash
cp env.example .env
nano .env
```

### تنظیم Nginx (اختیاری)

```bash
sudo apt install nginx
sudo cp nginx/nginx.conf /etc/nginx/nginx.conf
sudo systemctl restart nginx
```

### SSL Certificate (اختیاری)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 📊 مانیتورینگ و نگهداری

### بررسی وضعیت سرویس‌ها
```bash
# Docker
docker-compose ps

# PM2 (اگر از روش دستی استفاده کردید)
pm2 status

# Logs
docker-compose logs -f [service-name]
```

### بروزرسانی برنامه
```bash
# Docker
docker-compose pull
docker-compose up -d

# Manual
git pull
npm install
npm run build
pm2 restart all
```

### Backup دیتابیس
```bash
# PostgreSQL backup
pg_dump -U postgres dentalai > backup.sql

# Restore
psql -U postgres dentalai < backup.sql
```

---

## 🔍 عیب‌یابی

### مشکلات رایج

#### ۱. خطای اتصال به دیتابیس
```bash
# بررسی اتصال
psql -h localhost -U postgres -d dentalai

# بررسی Docker network
docker network ls
```

#### ۲. مشکل در AI Server
```bash
# بررسی پورت
netstat -tlnp | grep 8000

# چک کردن مدل‌ها
ls -la models/
```

#### ۳. خطای Build
```bash
# پاک کردن cache
npm run clean:all
rm -rf node_modules
npm install
```

#### ۴. مشکلات حافظه
```bash
# بررسی استفاده از RAM
free -h

# افزایش swap (اگر RAM کم است)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📞 پشتیبانی

اگر با مشکل مواجه شدید:

1. لاگ‌های Docker را چک کنید: `docker-compose logs`
2. لاگ‌های برنامه را ببینید
3. تنظیمات فایروال را بررسی کنید
4. اتصال شبکه بین کانتینرها را تست کنید

### پورت‌های مورد استفاده
- **80/443**: Nginx (اختیاری)
- **3000**: Dashboard
- **3030**: Frontend
- **5001**: AI Server (Python)
- **5432**: PostgreSQL
- **7272**: API (Backend)

---

## 🎯 چک‌لیست نهایی

- [ ] Docker و Docker Compose نصب هستند
- [ ] پورت‌های مورد نیاز باز هستند
- [ ] متغیرهای محیطی تنظیم شده‌اند
- [ ] مدل‌های AI دانلود شده‌اند
- [ ] دیتابیس migrate شده است
- [ ] SSL certificate تنظیم شده (اختیاری)
- [ ] Backup تنظیم شده است

**موفق باشید! 🦷🤖**
