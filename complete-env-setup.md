# راهنمای کامل تنظیم فایل .env برای DentalAI با Prisma و PostgreSQL

## 📋 مرحله ۱: نصب و تنظیم PostgreSQL

### نصب PostgreSQL روی Ubuntu:

```bash
# بروزرسانی سیستم
sudo apt update

# نصب PostgreSQL
sudo apt install postgresql postgresql-contrib

# بررسی وضعیت سرویس
sudo systemctl status postgresql
sudo systemctl enable postgresql
```

### ایجاد دیتابیس و کاربر:

```bash
# ورود به PostgreSQL
sudo -u postgres psql

# در محیط PostgreSQL:
```

```sql
-- ایجاد دیتابیس
CREATE DATABASE dentalai;

-- ایجاد کاربر (username و password را خودتان انتخاب کنید)
CREATE USER dentalai_user WITH PASSWORD 'DentalAI2024!Secure';

-- اعطای دسترسی کامل به کاربر
GRANT ALL PRIVILEGES ON DATABASE dentalai TO dentalai_user;

-- برای PostgreSQL 15+ (اگر خطا داد)
ALTER DATABASE dentalai OWNER TO dentalai_user;

-- خروج
\q
```

### تست اتصال:

```bash
# تست اتصال با کاربر جدید
psql -h localhost -U dentalai_user -d dentalai

# اگر خطا داد، تنظیمات authentication را تغییر دهید
sudo nano /etc/postgresql/*/main/pg_hba.conf
```

**در فایل `pg_hba.conf` این خط را اضافه کنید:**
```
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   dentalai        dentalai_user                           md5
host    dentalai        dentalai_user   127.0.0.1/32            md5
host    dentalai        dentalai_user   ::1/128                 md5
```

```bash
# راه‌اندازی مجدد PostgreSQL
sudo systemctl restart postgresql

# تست مجدد
psql -h localhost -U dentalai_user -d dentalai
```

---

## 📝 مرحله ۲: تغییر Prisma Schema برای PostgreSQL

### ویرایش فایل `minimal-api-dev-v6/prisma/schema.prisma`:

```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

// ... بقیه مدل‌ها بدون تغییر
```

---

## 🔐 مرحله ۳: فایل .env کامل

### فایل `.env` برای `/home/salahk/.env`:

```bash
# ============================================================================
# DentalAI - Environment Configuration
# ============================================================================

# ----------------------------------------------------------------------------
# Database Configuration (PostgreSQL)
# ----------------------------------------------------------------------------
# فرمت: postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE
# 
# مثال با اطلاعات شما:
# - Username: dentalai_user
# - Password: DentalAI2024!Secure (یا هر password که انتخاب کردید)
# - Host: localhost (یا 127.0.0.1)
# - Port: 5432 (پورت پیش‌فرض PostgreSQL)
# - Database: dentalai

DATABASE_URL="postgresql://dentalai_user:DentalAI2024!Secure@localhost:5432/dentalai?schema=public"

# برای اتصال از خارج (اگر نیاز دارید):
# DATABASE_URL="postgresql://dentalai_user:DentalAI2024!Secure@195.206.234.48:5432/dentalai?schema=public"

# ----------------------------------------------------------------------------
# Authentication & Security
# ----------------------------------------------------------------------------
# این secret key را تغییر دهید! از یک رشته تصادفی قوی استفاده کنید
# می‌توانید با این دستور یک secret قوی بسازید:
# openssl rand -base64 32

NEXTAUTH_SECRET="dentalai-nextauth-secret-key-change-this-to-random-string-$(date +%s)"
NEXTAUTH_URL="https://ceph.bioritalin.ir"

# JWT Secret برای احراز هویت
JWT_SECRET="dentalai-jwt-secret-key-change-this-$(date +%s)"

# Bcrypt rounds برای hash کردن رمز عبور
BCRYPT_ROUNDS=12

# ----------------------------------------------------------------------------
# API URLs
# ----------------------------------------------------------------------------
# استفاده از دامنه واقعی برای production
VITE_API_URL="https://ceph.bioritalin.ir"
VITE_AI_API_URL="https://ceph.bioritalin.ir"
NEXT_PUBLIC_API_URL="https://ceph.bioritalin.ir"

# برای development (اگر نیاز دارید):
# VITE_API_URL="http://localhost:7272"
# VITE_AI_API_URL="http://localhost:5001"
# NEXT_PUBLIC_API_URL="http://localhost:7272"

# ----------------------------------------------------------------------------
# Application Settings
# ----------------------------------------------------------------------------
NODE_ENV="production"
FLASK_ENV="production"
PORT=7272

# ----------------------------------------------------------------------------
# Python AI Server Settings
# ----------------------------------------------------------------------------
PYTHONPATH="/home/salahk"

# CPU Optimization برای AI Server
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
VECLIB_MAXIMUM_THREADS=2
NUMEXPR_NUM_THREADS=2

# غیرفعال کردن CUDA (استفاده از CPU)
CUDA_VISIBLE_DEVICES=""

# ----------------------------------------------------------------------------
# File Upload Settings
# ----------------------------------------------------------------------------
UPLOAD_DIR="/home/salahk/uploads"
MAX_FILE_SIZE=104857600  # 100MB in bytes

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
LOG_LEVEL="INFO"
LOG_DIR="/home/salahk/logs"

# ----------------------------------------------------------------------------
# External Services (اگر استفاده می‌کنید)
# ----------------------------------------------------------------------------
# OpenAI API (اگر از OpenAI استفاده می‌کنید)
# OPENAI_API_KEY="your-openai-api-key"

# Supabase (اگر از Supabase استفاده می‌کنید)
# SUPABASE_URL="your-supabase-url"
# SUPABASE_ANON_KEY="your-supabase-anon-key"

# ----------------------------------------------------------------------------
# Email Settings (اگر از email استفاده می‌کنید)
# ----------------------------------------------------------------------------
# SMTP_HOST="smtp.gmail.com"
# SMTP_PORT=587
# SMTP_USER="your-email@gmail.com"
# SMTP_PASSWORD="your-app-password"
# SMTP_FROM="noreply@ceph.bioritalin.ir"

# ----------------------------------------------------------------------------
# Payment Gateway (اگر از درگاه پرداخت استفاده می‌کنید)
# ----------------------------------------------------------------------------
# ZARINPAL_MERCHANT_ID="your-merchant-id"
# NOWPAYMENTS_API_KEY="your-nowpayments-key"
```

---

## 🔧 مرحله ۴: تنظیم Prisma

### روی سرور:

```bash
cd /home/salahk/backend

# نصب وابستگی‌ها
npm install

# Generate Prisma Client
npx prisma generate

# اجرای migrations برای ایجاد جداول
npx prisma db push

# یا اگر migrations دارید:
# npx prisma migrate deploy

# بررسی اتصال به دیتابیس
npx prisma db pull

# مشاهده دیتابیس در Prisma Studio (اختیاری)
npx prisma studio
```

---

## 🔐 نکات امنیتی برای Password:

### انتخاب Password قوی:

```bash
# روش ۱: استفاده از openssl
openssl rand -base64 32

# روش ۲: استفاده از pwgen (اگر نصب است)
pwgen -s 32 1

# روش ۳: استفاده از /dev/urandom
cat /dev/urandom | tr -dc 'a-zA-Z0-9!@#$%^&*' | fold -w 32 | head -n 1
```

### مثال Password امن:
```
DentalAI2024!Secure@PostgreSQL#Random
```

---

## 📋 چک‌لیست تنظیمات:

### PostgreSQL:
- [ ] PostgreSQL نصب شده
- [ ] دیتابیس `dentalai` ایجاد شده
- [ ] کاربر `dentalai_user` ایجاد شده
- [ ] Password قوی تنظیم شده
- [ ] دسترسی‌ها اعطا شده
- [ ] اتصال تست شده

### Prisma:
- [ ] `schema.prisma` برای PostgreSQL تغییر یافته
- [ ] `DATABASE_URL` در `.env` تنظیم شده
- [ ] Prisma Client generate شده
- [ ] Migrations اجرا شده
- [ ] جداول در دیتابیس ایجاد شده

### Environment Variables:
- [ ] فایل `.env` ایجاد شده
- [ ] همه متغیرها تنظیم شده
- [ ] Secret keys تغییر یافته
- [ ] API URLs با دامنه واقعی تنظیم شده

---

## 🚨 عیب‌یابی مشکلات رایج:

### مشکل ۱: خطای اتصال به دیتابیس

```bash
# بررسی وضعیت PostgreSQL
sudo systemctl status postgresql

# بررسی پورت
netstat -tlnp | grep 5432

# تست اتصال
psql -h localhost -U dentalai_user -d dentalai
```

### مشکل ۲: خطای Prisma

```bash
# پاک کردن Prisma Client
rm -rf node_modules/.prisma

# Generate مجدد
npx prisma generate

# بررسی schema
npx prisma validate
```

### مشکل ۳: خطای Migration

```bash
# Reset دیتابیس (⚠️ فقط برای development!)
npx prisma migrate reset

# یا push مستقیم
npx prisma db push --force-reset
```

---

## ✅ نتیجه:

پس از تکمیل این مراحل:

1. ✅ PostgreSQL نصب و تنظیم شده
2. ✅ دیتابیس و کاربر ایجاد شده
3. ✅ Prisma برای PostgreSQL تنظیم شده
4. ✅ فایل `.env` کامل تنظیم شده
5. ✅ جداول در دیتابیس ایجاد شده

**پروژه شما آماده استفاده از PostgreSQL است! 🚀**



