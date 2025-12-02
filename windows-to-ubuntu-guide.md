# راهنمای انتقال پروژه از ویندوز به Ubuntu Server

## 📋 مراحل کامل انتقال و اجرای پروژه

### مرحله ۱: آماده‌سازی پروژه در ویندوز

```powershell
# ابتدا پروژه را روی ویندوز build کنید
cd vite-js
npm install
npm run build

# بررسی کنید که فایل‌های build شده در dist/ وجود دارند
dir dist\
```

### مرحله ۲: انتقال فایل‌ها به سرور Ubuntu

#### گزینه ۱: استفاده از SCP (پیشنهادی - رایگان)

```bash
# روی ویندوز PowerShell:
scp -r "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\vite-js" user@your-server-ip:/home/user/
```

#### گزینه ۲: استفاده از WinSCP (رابط گرافیکی)

1. دانلود WinSCP از https://winscp.net/
2. اتصال به سرور Ubuntu با اطلاعات SSH
3. انتقال پوشه `vite-js` به سرور

#### گزینه ۳: استفاده از Git (اگر پروژه روی Git باشد)

```bash
# روی سرور Ubuntu:
git clone your-repository-url
cd your-project/vite-js
```

### مرحله ۳: آماده‌سازی سرور Ubuntu

```bash
# اتصال به سرور
ssh user@your-server-ip

# بروزرسانی سیستم
sudo apt update && sudo apt upgrade -y

# نصب Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# بررسی نصب
node --version
npm --version
```

### مرحله ۴: Build و اجرای پروژه روی سرور

```bash
# رفتن به دایرکتوری پروژه
cd ~/vite-js

# نصب وابستگی‌ها
npm install

# اگر فایل‌های build شده را انتقال دادید، حذف و دوباره build کنید
rm -rf dist/
npm run build

# بررسی فایل‌های ساخته شده
ls -la dist/
```

### مرحله ۵: راه‌اندازی وب‌سرور

#### گزینه ۱: استفاده از Nginx (توصیه شده برای production)

```bash
# نصب Nginx
sudo apt install nginx

# ایجاد دایرکتوری برای پروژه
sudo mkdir -p /var/www/dentalai

# کپی فایل‌های build شده
sudo cp -r dist/* /var/www/dentalai/

# تنظیم دسترسی‌ها
sudo chown -R www-data:www-data /var/www/dentalai

# ایجاد فایل تنظیمات Nginx
sudo nano /etc/nginx/sites-available/dentalai
```

محتوای فایل تنظیمات Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # یا IP سرور
    root /var/www/dentalai;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # بهینه‌سازی کش برای فایل‌های static
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # فعال‌سازی gzip
    gzip on;
    gzip_types text/css application/javascript text/javascript application/json;
}
```

```bash
# فعال‌سازی سایت
sudo ln -s /etc/nginx/sites-available/dentalai /etc/nginx/sites-enabled/

# غیرفعال‌سازی سایت پیش‌فرض
sudo unlink /etc/nginx/sites-enabled/default

# تست تنظیمات
sudo nginx -t

# راه‌اندازی Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

#### گزینه ۲: استفاده از PM2 (برای توسعه)

```bash
# نصب PM2
sudo npm install -g pm2

# از دایرکتوری پروژه
cd ~/vite-js

# راه‌اندازی سرور توسعه
pm2 start "npm run dev -- --host 0.0.0.0" --name dentalai

# یا برای فایل‌های build شده
npx serve -s dist -l 3000
pm2 start "npx serve -s dist -l 3000" --name dentalai-prod
```

### مرحله ۶: تنظیم فایروال و دسترسی

```bash
# باز کردن پورت 80 برای HTTP
sudo ufw allow 80
sudo ufw allow 22  # برای SSH
sudo ufw --force enable

# بررسی وضعیت
sudo ufw status
```

## 🔧 عیب‌یابی مشکلات رایج

### مشکل ۱: خطای build
```bash
# پاک کردن node_modules و دوباره نصب
rm -rf node_modules package-lock.json
npm install
npm run build
```

### مشکل ۲: خطای permission در Nginx
```bash
# تنظیم دسترسی صحیح
sudo chown -R www-data:www-data /var/www/dentalai
sudo chmod -R 755 /var/www/dentalai
```

### مشکل ۳: پورت 80 اشغال است
```bash
# بررسی چه برنامه‌ای از پورت استفاده می‌کند
sudo netstat -tlnp | grep :80

# یا استفاده از پورت دیگر در Nginx
# تغییر listen 80 به listen 8080 در تنظیمات Nginx
```

## 📊 بررسی عملکرد

```bash
# بررسی وضعیت Nginx
sudo systemctl status nginx

# مشاهده لاگ‌های Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# بررسی وضعیت PM2 (اگر استفاده می‌کنید)
pm2 status
pm2 logs dentalai
```

## 🎯 نتیجه نهایی

پس از تکمیل مراحل بالا، پروژه شما روی آدرس زیر قابل دسترسی خواهد بود:
- **با Nginx**: `http://your-server-ip`
- **با PM2**: `http://your-server-ip:3000` (یا پورتی که تنظیم کرده‌اید)

## 💡 نکات مهم

1. **SSL**: برای محیط production از Let's Encrypt برای HTTPS استفاده کنید
2. **بکاپ**: همیشه از فایل‌های مهم بکاپ بگیرید
3. **مانیتورینگ**: از ابزارهایی مانند htop برای مانیتورینگ سرور استفاده کنید
4. **امنیت**: رمز عبور قوی برای SSH استفاده کنید و فایروال را تنظیم کنید

موفق باشید! 🚀



