# راهنمای انتقال دستی فایل‌ها به سرور

## 🎯 روش‌های انتقال فایل‌ها

### روش ۱: استفاده از WinSCP (ساده‌ترین)

1. **دانلود WinSCP**: https://winscp.net/
2. **باز کردن WinSCP**
3. **تنظیمات اتصال**:
   - Host name: `195.206.234.48`
   - User name: `root`
   - Password: رمز عبور سرور
4. **اتصال**
5. **انتقال فایل‌ها**:

#### سمت چپ (ویندوز):
```
C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\vite-js\dist\
```

#### سمت راست (سرور):
```
/home/root/dentalai/frontend/
```

### فایل‌های مورد نیاز:

#### ۱. Frontend:
```
از: C:\...\vite-js\dist\
به: /home/root/dentalai/frontend/
```

#### ۲. Backend:
```
از: C:\...\minimal-api-dev-v6\.next\
به: /home/root/dentalai/backend/.next/

از: C:\...\minimal-api-dev-v6\package.json
به: /home/root/dentalai/backend/

از: C:\...\minimal-api-dev-v6\next.config.mjs
به: /home/root/dentalai/backend/

از: C:\...\minimal-api-dev-v6\prisma\
به: /home/root/dentalai/backend/prisma/
```

#### ۳. AI Server:
```
از: C:\...\unified_ai_api_server.py
به: /home/root/dentalai/

از: C:\...\requirements_unified_api.txt
به: /home/root/dentalai/

از: C:\...\cephx_service\
به: /home/root/dentalai/cephx_service\

از: C:\...\facial-landmark-detection\
به: /home/root/dentalai/facial-landmark-detection\

از: C:\...\CLdetection2023\
به: /home/root/dentalai/CLdetection2023\
```

#### ۴. تنظیمات:
```
از: C:\...\env.example
به: /home/root/dentalai/

از: C:\...\docker-compose.yml
به: /home/root/dentalai/

از: C:\...\nginx\
به: /home/root/dentalai/nginx\

از: C:\...\quick-start.sh
به: /home/root/dentalai/
```

## 🔧 روش ۲: استفاده از PowerShell/Command Prompt

```cmd
REM انتقال Frontend
scp -r "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\vite-js\dist" root@195.206.234.48:/home/root/dentalai/frontend/

REM انتقال Backend
scp -r "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\minimal-api-dev-v6\.next" root@195.206.234.48:/home/root/dentalai/backend/
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\minimal-api-dev-v6\package.json" root@195.206.234.48:/home/root/dentalai/backend/
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\minimal-api-dev-v6\next.config.mjs" root@195.206.234.48:/home/root/dentalai/backend/
scp -r "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\minimal-api-dev-v6\prisma" root@195.206.234.48:/home/root/dentalai/backend/

REM انتقال AI Server
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\unified_ai_api_server.py" root@195.206.234.48:/home/root/dentalai/
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\requirements_unified_api.txt" root@195.206.234.48:/home/root/dentalai/

REM انتقال تنظیمات
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\env.example" root@195.206.234.48:/home/root/dentalai/
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\docker-compose.yml" root@195.206.234.48:/home/root/dentalai/
scp -r "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\nginx" root@195.206.234.48:/home/root/dentalai/
scp "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\quick-start.sh" root@195.206.234.48:/home/root/dentalai/
```

## 📋 چک‌لیست انتقال:

### Frontend:
- [ ] `vite-js/dist/` انتقال داده شده به `frontend/`

### Backend:
- [ ] `minimal-api-dev-v6/.next/` انتقال داده شده به `backend/.next/`
- [ ] `package.json` انتقال داده شده به `backend/`
- [ ] `next.config.mjs` انتقال داده شده به `backend/`
- [ ] `prisma/` انتقال داده شده به `backend/prisma/`

### AI Server:
- [ ] `unified_ai_api_server.py` انتقال داده شده
- [ ] `requirements_unified_api.txt` انتقال داده شده
- [ ] فولدرهای `cephx_service/`, `facial-landmark-detection/`, `CLdetection2023/` انتقال داده شده

### تنظیمات:
- [ ] `env.example` انتقال داده شده و به `.env` تغییر نام داده شده
- [ ] `docker-compose.yml` انتقال داده شده
- [ ] `nginx/` انتقال داده شده
- [ ] `quick-start.sh` انتقال داده شده

## 🚀 راه‌اندازی روی سرور:

```bash
# روی سرور Ubuntu
cd /home/root/dentalai

# تنظیم متغیرهای محیطی
cp env.example .env
nano .env  # تنظیم مقادیر

# راه‌اندازی
docker-compose up -d

# یا استفاده از اسکریپت
chmod +x quick-start.sh
./quick-start.sh
```

## 🌐 دسترسی:

- Frontend: http://195.206.234.48:3030
- API: http://195.206.234.48:7272
- AI Server: http://195.206.234.48:5000



