# راهنمای Deployment پروژه DentalAI با OLSPanel

## 📋 معرفی OLSPanel

[OLSPanel](https://olspanel.com/) یک کنترل پنل رایگان و سبک برای سرورهای OpenLiteSpeed است که شامل ویژگی‌های زیر می‌شود:

### ✅ مزایای OLSPanel برای پروژه DentalAI:
- **رایگان و سبک**: حداقل منابع (1GB RAM، ۱ CPU core، ۱۰GB disk)
- **پشتیبانی از چندین تکنولوژی**: PHP، Node.js، Python، Static sites
- **پشتیبانی از PostgreSQL**: برای دیتابیس Prisma
- **Auto SSL**: گواهی SSL رایگان از Let's Encrypt
- **امنیت بالا**: Firewall، IP blocking، ۲FA
- **مدیریت آسان**: File manager، backup، cron jobs

---

## 🚀 مراحل راه‌اندازی با OLSPanel

### مرحله ۱: نصب OLSPanel روی سرور Ubuntu

```bash
# اجرای اسکریپت نصب OLSPanel
bash <(curl -fsSL https://olspanel.com/install.sh || wget -qO- https://olspanel.com/install.sh)
```

پس از نصب، اطلاعات login نمایش داده می‌شود:
- **Username**: نمایش داده می‌شود
- **Password**: نمایش داده می‌شود
- **Port**: نمایش داده می‌شود

### مرحله ۲: دسترسی به پنل مدیریت

1. مرورگر را باز کنید
2. آدرس: `http://your-server-ip:port`
3. با username و password وارد شوید

### مرحله ۳: تنظیمات اولیه سرور

#### نصب PostgreSQL
```bash
# از طریق SSH یا terminal سرور
sudo apt update
sudo apt install postgresql postgresql-contrib

# ایجاد دیتابیس و کاربر
sudo -u postgres psql
CREATE DATABASE dentalai;
CREATE USER dentalai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE dentalai TO dentalai_user;
\q
```

#### نصب Node.js و Python
```bash
# Node.js (اگر نصب نشده)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Python و pip
sudo apt install python3 python3-pip python3-venv
```

---

## 🌐 راه‌اندازی برنامه‌ها در OLSPanel

### ۱. تنظیم Frontend (React/Vite)

#### از طریق پنل OLSPanel:
1. **Sites** → **Add Site**
2. **Application Type**: `Node.js`
3. **Domain**: `your-domain.com` یا `frontend.your-domain.com`
4. **Node.js Version**: `18` یا بالاتر
5. **Application Path**: `/home/user/dentalai/vite-js`
6. **Start Command**: `npm run dev -- --host 0.0.0.0 --port 3030`
7. **Port**: `3030`

#### تنظیمات Environment:
```bash
# در تنظیمات Site
NODE_ENV=production
VITE_API_URL=http://localhost:7272
VITE_AI_API_URL=http://localhost:5000
```

### ۲. تنظیم Backend API (Next.js)

#### از طریق پنل OLSPanel:
1. **Sites** → **Add Site**
2. **Application Type**: `Node.js`
3. **Domain**: `api.your-domain.com`
4. **Node.js Version**: `18`
5. **Application Path**: `/home/user/dentalai/minimal-api-dev-v6`
6. **Start Command**: `npm run dev -- -p 7272`
7. **Port**: `7272`

#### تنظیمات Environment:
```bash
DATABASE_URL=postgresql://dentalai_user:your_password@localhost:5432/dentalai
NEXTAUTH_SECRET=your-secret-key
NEXTAUTH_URL=http://api.your-domain.com
NODE_ENV=production
```

### ۳. تنظیم AI Server (Python)

#### از طریق پنل OLSPanel:
1. **Sites** → **Add Site**
2. **Application Type**: `Python`
3. **Domain**: `ai.your-domain.com`
4. **Python Version**: `3.x`
5. **Application Path**: `/home/user/dentalai`
6. **Start Command**: `python unified_ai_api_server.py`
7. **Port**: `5000`

#### تنظیمات Environment:
```bash
PYTHONPATH=/home/user/dentalai
FLASK_ENV=production
```

### ۴. تنظیم Dashboard (Next.js)

#### از طریق پنل OLSPanel:
1. **Sites** → **Add Site**
2. **Application Type**: `Node.js`
3. **Domain**: `dashboard.your-domain.com`
4. **Node.js Version**: `18`
5. **Application Path**: `/home/user/dentalai/next-js`
6. **Start Command**: `npm run start`
7. **Port**: `3000`

---

## 🔧 تنظیمات پیشرفته

### تنظیم Reverse Proxy (اختیاری)

اگر می‌خواهید همه سرویس‌ها زیر یک دامنه باشند:

#### فایل تنظیمات Nginx در OLSPanel:
```nginx
# در تنظیمات Site اصلی
location /api/ {
    proxy_pass http://localhost:7272/api/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

location /ai-api/ {
    proxy_pass http://localhost:5000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

location /dashboard/ {
    proxy_pass http://localhost:3000/dashboard/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### تنظیم SSL Certificate

#### از طریق OLSPanel:
1. **Sites** → انتخاب site
2. **SSL** → **Free SSL**
3. دامنه را وارد کنید
4. **Install SSL**

### تنظیم Firewall

#### از طریق OLSPanel:
1. **Security** → **Firewall**
2. پورت‌های مورد نیاز را باز کنید:
   - 3030 (Frontend)
   - 7272 (API)
   - 5000 (AI Server)
   - 3000 (Dashboard)
   - 80, 443 (HTTP/HTTPS)

---

## 📊 مانیتورینگ و نگهداری

### بررسی وضعیت سرویس‌ها

#### از طریق پنل OLSPanel:
1. **Sites** → مشاهده وضعیت هر site
2. **Resource Monitor** → مصرف منابع
3. **Logs** → لاگ‌های هر سرویس

### Backup و Restore

#### از طریق OLSPanel:
1. **Backup** → **Create Backup**
2. انتخاب فایل‌ها و دیتابیس
3. زمان‌بندی خودکار backup

### بروزرسانی برنامه

```bash
# روی سرور
cd /home/user/dentalai

# بروزرسانی کد
git pull

# بروزرسانی وابستگی‌ها
cd vite-js && npm install
cd ../minimal-api-dev-v6 && npm install
cd ../next-js && npm install

# راه‌اندازی مجدد سرویس‌ها از طریق پنل OLSPanel
```

---

## 🔍 عیب‌یابی مشکلات رایج

### مشکل اتصال به دیتابیس
```bash
# بررسی اتصال
psql -h localhost -U dentalai_user -d dentalai

# بررسی تنظیمات Prisma
cd minimal-api-dev-v6
npx prisma generate
```

### مشکل پورت‌ها
```bash
# بررسی پورت‌های باز
netstat -tlnp | grep -E ':(3030|7272|5000|3000)'

# اگر پورت اشغال است
sudo fuser -k 3030/tcp  # یا پورت دیگر
```

### مشکل حافظه
```bash
# بررسی مصرف RAM
free -h

# اگر کم است، swap اضافه کنید
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📈 مزایای استفاده از OLSPanel

### ⚡ عملکرد بالا
- OpenLiteSpeed سرور سریع
- بهینه‌سازی خودکار
- مصرف منابع کم

### 🔒 امنیت
- Firewall پیشرفته
- SSL خودکار
- IP blocking
- ۲FA

### 🎛️ مدیریت آسان
- رابط گرافیکی کاربرپسند
- مدیریت یک‌کلیک
- مانیتورینگ real-time

### 💰 رایگان
- بدون هزینه
- تمام ویژگی‌های ضروری
- جامعه فعال

---

## 🎯 نتیجه‌گیری

OLSPanel انتخاب عالی برای پروژه DentalAI است چون:

1. **رایگان و سبک**: مناسب برای شروع
2. **پشتیبانی کامل**: Node.js، Python، PostgreSQL
3. **مدیریت آسان**: رابط گرافیکی برای همه تنظیمات
4. **امنیت بالا**: firewall، SSL، backup
5. **انعطاف**: امکان تنظیمات پیشرفته

با OLSPanel می‌توانید پروژه DentalAI را به راحتی روی سرور راه‌اندازی و مدیریت کنید! 🚀

**نکته:** برای production، از حداقل ۲GB RAM استفاده کنید.



