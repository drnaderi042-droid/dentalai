# راهنمای تغییر Prisma از SQLite به PostgreSQL

## 📋 مرحله ۱: تغییر Prisma Schema

### ویرایش فایل `minimal-api-dev-v6/prisma/schema.prisma`:

**قبل (SQLite):**
```prisma
datasource db {
  provider = "sqlite"
  url      = "file:./dev.db"
}
```

**بعد (PostgreSQL):**
```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}
```

---

## 🔧 مرحله ۲: تنظیم DATABASE_URL در .env

### فرمت DATABASE_URL:

```
postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE?schema=public
```

### مثال با اطلاعات شما:

```bash
# اگر username: dentalai_user
# اگر password: DentalAI2024!Secure
# Host: localhost
# Port: 5432
# Database: dentalai

DATABASE_URL="postgresql://dentalai_user:DentalAI2024!Secure@localhost:5432/dentalai?schema=public"
```

### نکات مهم:

1. **Username و Password**: همان مقادیری که در PostgreSQL ایجاد کردید
2. **Host**: `localhost` برای اتصال محلی، یا IP سرور برای اتصال از راه دور
3. **Port**: `5432` پورت پیش‌فرض PostgreSQL
4. **Database**: `dentalai` نام دیتابیسی که ایجاد کردید
5. **schema**: `public` schema پیش‌فرض PostgreSQL

---

## 🚀 مرحله ۳: اجرای Prisma Commands

### روی سرور:

```bash
cd /home/salahk/backend

# ۱. نصب وابستگی‌ها (اگر نصب نشده)
npm install

# ۲. Generate Prisma Client
npx prisma generate

# ۳. Push schema به دیتابیس (ایجاد جداول)
npx prisma db push

# ۴. یا اگر migrations دارید:
npx prisma migrate deploy
```

---

## 📊 مرحله ۴: بررسی و تست

### بررسی جداول:

```bash
# مشاهده جداول در دیتابیس
psql -h localhost -U dentalai_user -d dentalai -c "\dt"

# یا استفاده از Prisma Studio
npx prisma studio
# سپس در مرورگر: http://localhost:5555
```

---

## 🔐 ایجاد Username و Password در PostgreSQL

### روش ۱: استفاده از اسکریپت خودکار

```bash
# اجرای اسکریپت تنظیم دیتابیس
chmod +x setup-database.sh
./setup-database.sh
```

این اسکریپت به طور خودکار:
- PostgreSQL را نصب می‌کند
- دیتابیس و کاربر ایجاد می‌کند
- Password تصادفی قوی تولید می‌کند
- فایل `.env` را با اطلاعات صحیح ایجاد می‌کند

### روش ۲: دستی

```bash
# ورود به PostgreSQL
sudo -u postgres psql

# در محیط PostgreSQL:
```

```sql
-- ایجاد دیتابیس
CREATE DATABASE dentalai;

-- ایجاد کاربر با password
CREATE USER dentalai_user WITH PASSWORD 'DentalAI2024!Secure';

-- اعطای دسترسی
GRANT ALL PRIVILEGES ON DATABASE dentalai TO dentalai_user;

-- برای PostgreSQL 15+
ALTER DATABASE dentalai OWNER TO dentalai_user;

-- خروج
\q
```

### تست اتصال:

```bash
# تست با password
psql -h localhost -U dentalai_user -d dentalai
# Password را وارد کنید: DentalAI2024!Secure
```

---

## 📝 فایل .env کامل

```bash
# Database
DATABASE_URL="postgresql://dentalai_user:DentalAI2024!Secure@localhost:5432/dentalai?schema=public"

# Authentication
NEXTAUTH_SECRET="your-secret-key-here"
NEXTAUTH_URL="https://ceph.bioritalin.ir"
JWT_SECRET="your-jwt-secret-here"
BCRYPT_ROUNDS=12

# API URLs
VITE_API_URL="https://ceph.bioritalin.ir"
VITE_AI_API_URL="https://ceph.bioritalin.ir"
NEXT_PUBLIC_API_URL="https://ceph.bioritalin.ir"

# Application
NODE_ENV="production"
FLASK_ENV="production"
PORT=7272

# Python AI Server
PYTHONPATH="/home/salahk"
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=""

# File Upload
UPLOAD_DIR="/home/salahk/uploads"
MAX_FILE_SIZE=104857600

# Logging
LOG_LEVEL="INFO"
LOG_DIR="/home/salahk/logs"
```

---

## ✅ چک‌لیست نهایی:

- [ ] PostgreSQL نصب شده
- [ ] دیتابیس `dentalai` ایجاد شده
- [ ] کاربر `dentalai_user` ایجاد شده
- [ ] Password تنظیم شده
- [ ] `schema.prisma` برای PostgreSQL تغییر یافته
- [ ] `DATABASE_URL` در `.env` تنظیم شده
- [ ] Prisma Client generate شده
- [ ] جداول در دیتابیس ایجاد شده
- [ ] اتصال تست شده

---

## 🎯 نتیجه:

پس از تکمیل این مراحل، Prisma شما با PostgreSQL کار می‌کند و می‌توانید از دیتابیس استفاده کنید!



