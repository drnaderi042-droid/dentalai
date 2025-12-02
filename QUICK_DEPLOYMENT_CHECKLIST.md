# ✅ چک‌لیست سریع Deployment
# Quick Deployment Checklist

این چک‌لیست برای مرور سریع مراحل deployment است.

## 📋 قبل از شروع

- [ ] سرور با مشخصات مناسب آماده است
- [ ] Domain name خریداری و تنظیم شده است
- [ ] DNS records به درستی تنظیم شده‌اند
- [ ] دسترسی SSH به سرور دارید
- [ ] فایل‌های پروژه آماده آپلود هستند

## 🔧 نصب و راه‌اندازی

### 1. سرور
- [ ] سیستم عامل به‌روزرسانی شده
- [ ] Node.js 18+ نصب شده
- [ ] Python 3.8+ نصب شده
- [ ] PostgreSQL نصب شده
- [ ] Nginx نصب شده
- [ ] PM2 نصب شده
- [ ] Firewall تنظیم شده

### 2. پروژه
- [ ] فایل‌های پروژه آپلود شده‌اند
- [ ] Frontend build شده (`npm run build`)
- [ ] Backend build شده (`npm run build`)
- [ ] Python virtual environment ایجاد شده
- [ ] Python dependencies نصب شده

### 3. Database
- [ ] Database ایجاد شده
- [ ] User و password تنظیم شده
- [ ] Migrations اجرا شده
- [ ] Connection تست شده

### 4. Configuration
- [ ] Environment variables تنظیم شده
- [ ] JWT_SECRET تغییر یافته
- [ ] Database credentials تنظیم شده
- [ ] AI_SERVER_URL تنظیم شده
- [ ] CORS origins تنظیم شده

### 5. Services
- [ ] Backend API با PM2 اجرا می‌شود
- [ ] Python AI Server با systemd اجرا می‌شود
- [ ] Nginx configuration تست شده
- [ ] SSL certificate نصب شده (HTTPS)

## 🔒 امنیت

- [ ] Passwords پیش‌فرض تغییر یافته
- [ ] JWT_SECRET قوی تنظیم شده
- [ ] File permissions صحیح تنظیم شده
- [ ] Firewall فعال است
- [ ] فقط پورت‌های لازم باز هستند
- [ ] CORS به درستی تنظیم شده
- [ ] Rate limiting فعال است

## 📊 Monitoring

- [ ] PM2 monitoring فعال است
- [ ] Log files در دسترس هستند
- [ ] Health check endpoints کار می‌کنند
- [ ] Backup script تنظیم شده
- [ ] Cron job برای backup تنظیم شده

## ✅ تست نهایی

- [ ] Frontend در دسترس است
- [ ] Backend API پاسخ می‌دهد
- [ ] Python AI Server کار می‌کند
- [ ] Database connection برقرار است
- [ ] File upload کار می‌کند
- [ ] AI endpoints پاسخ می‌دهند
- [ ] SSL certificate معتبر است
- [ ] Performance قابل قبول است

## 📝 Documentation

- [ ] API documentation ایجاد شده
- [ ] User guide آماده است
- [ ] Credentials در جای امن ذخیره شده
- [ ] Backup location مشخص است

---

## 🚨 مشکلات رایج

### Frontend نمایش داده نمی‌شود
- [ ] Nginx configuration صحیح است
- [ ] فایل‌های build در مسیر صحیح هستند
- [ ] Permissions صحیح هستند

### Backend API کار نمی‌کند
- [ ] PM2 process در حال اجرا است
- [ ] Port 7272 در دسترس است
- [ ] Database connection برقرار است
- [ ] Environment variables صحیح هستند

### Python AI Server کار نمی‌کند
- [ ] systemd service فعال است
- [ ] Virtual environment فعال است
- [ ] Dependencies نصب شده‌اند
- [ ] Port 5001 در دسترس است
- [ ] Model files در مسیر صحیح هستند

### Database Connection Error
- [ ] PostgreSQL در حال اجرا است
- [ ] User و password صحیح هستند
- [ ] Database وجود دارد
- [ ] Permissions صحیح هستند

---

## 📞 پشتیبانی

در صورت بروز مشکل:
1. لاگ‌ها را بررسی کنید
2. Health check endpoints را تست کنید
3. Documentation را مطالعه کنید
4. از DEPLOYMENT_GUIDE.md استفاده کنید

---

**موفق باشید! 🚀**

