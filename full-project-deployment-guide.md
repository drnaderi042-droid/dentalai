# راهنمای کامل Deployment پروژه DentalAI

## 📋 ساختار پروژه

پروژه DentalAI شامل چندین کامپوننت است:

### 🎨 Frontend Components
1. **vite-js/** - React/Vite Frontend (پورت 3030)
2. **next-js/** - Next.js Dashboard (پورت 3000)

### 🔧 Backend Components
3. **minimal-api-dev-v6/** - Next.js API با Prisma Database (پورت 7272)
4. **unified_ai_api_server.py** - Python AI/ML API Server (پورت 8000)

### 🤖 AI/ML Components
5. **اسکریپت‌های پایتون** - پردازش تصویر، مدل‌های ML، تحلیل‌های AI

---

## 🚀 استراتژی Deployment

### رویکرد پیشنهادی: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: dentalai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # AI/ML Python Server
  ai-api:
    build: ./python-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./datasets:/app/datasets
    environment:
      - PYTHONPATH=/app
    depends_on:
      - postgres

  # Next.js API (Prisma)
  api:
    build: ./minimal-api-dev-v6
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/dentalai
    depends_on:
      - postgres
      - ai-api

  # Main Frontend (Vite)
  frontend:
    build: ./vite-js
    ports:
      - "3030:80"
    depends_on:
      - api

  # Dashboard (Next.js)
  dashboard:
    build: ./next-js
    ports:
      - "3000:3000"
    depends_on:
      - api

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
      - dashboard
      - api

volumes:
  postgres_data:
```

---

## 📦 روش سریع: انتقال مرحله‌ای

### مرحله ۱: آماده‌سازی محیط سرور

```bash
# بروزرسانی سیستم
sudo apt update && sudo apt upgrade -y

# نصب Docker و Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install docker-compose-plugin

# نصب Node.js برای build
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# نصب Python برای AI server
sudo apt install python3 python3-pip python3-venv

# نصب PostgreSQL
sudo apt install postgresql postgresql-contrib
```

### مرحله ۲: انتقال فایل‌ها

```bash
# انتقال کل پروژه
scp -r "/path/to/project" user@server:/home/user/dentalai/

# یا انتقال مرحله‌ای
scp -r vite-js user@server:/home/user/
scp -r minimal-api-dev-v6 user@server:/home/user/
scp -r next-js user@server:/home/user/
scp unified_ai_api_server.py user@server:/home/user/
scp requirements_unified_api.txt user@server:/home/user/
```

### مرحله ۳: راه‌اندازی Database

```bash
# تنظیم PostgreSQL
sudo -u postgres psql
CREATE DATABASE dentalai;
CREATE USER dentalai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE dentalai TO dentalai_user;
\q

# یا با Docker
docker run -d \
  --name dentalai-postgres \
  -e POSTGRES_DB=dentalai \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15
```

### مرحله ۴: راه‌اندازی Python AI Server

```bash
# ایجاد محیط مجازی
python3 -m venv dentalai-env
source dentalai-env/bin/activate

# نصب وابستگی‌ها
pip install -r requirements_unified_api.txt

# دانلود مدل‌ها (در صورت نیاز)
python download_dlib_model.py
python download_sam_model.py

# راه‌اندازی سرور
python unified_ai_api_server.py
```

### مرحله ۵: راه‌اندازی Next.js API

```bash
cd minimal-api-dev-v6

# نصب وابستگی‌ها
npm install

# تنظیم Prisma
npx prisma generate
npx prisma db push

# راه‌اندازی API
npm run dev
```

### مرحله ۶: Build و راه‌اندازی Frontendها

```bash
# Build Vite Frontend
cd ../vite-js
npm install
npm run build

# Build Next.js Dashboard
cd ../next-js
npm install
npm run build
```

### مرحله ۷: تنظیم Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/dentalai
server {
    listen 80;
    server_name your-domain.com;

    # Main Frontend (Vite)
    location / {
        proxy_pass http://localhost:3030;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Dashboard
    location /dashboard {
        proxy_pass http://localhost:3000/dashboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api {
        proxy_pass http://localhost:7272/api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # AI API
    location /ai-api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔄 روش نیمه‌خودکار با PM2

```bash
# نصب PM2
sudo npm install -g pm2

# راه‌اندازی AI Server
pm2 start unified_ai_api_server.py --name "dentalai-ai" --interpreter python3

# راه‌اندازی API
cd minimal-api-dev-v6
pm2 start "npm run dev" --name "dentalai-api"

# راه‌اندازی Frontendها
cd ../vite-js
pm2 start "npm run preview" --name "dentalai-frontend"

cd ../next-js
pm2 start "npm run start" --name "dentalai-dashboard"

# ذخیره تنظیمات
pm2 save
pm2 startup
```

---

## 📊 منابع مورد نیاز سرور

| کامپوننت | RAM | CPU | Storage |
|----------|-----|-----|---------|
| PostgreSQL | 512MB | 1 core | 10GB |
| AI Python Server | 4GB | 4 cores | 20GB |
| Next.js API | 1GB | 2 cores | 5GB |
| Vite Frontend | 512MB | 1 core | 2GB |
| Next.js Dashboard | 1GB | 2 cores | 5GB |
| **مجموع** | **7GB+** | **4+ cores** | **42GB+** |

### تنظیمات پیشنهادی سرور:
- **RAM**: حداقل ۸GB
- **CPU**: حداقل ۴ هسته
- **Storage**: حداقل ۵۰GB SSD
- **OS**: Ubuntu 22.04 LTS

---

## 🚨 نکات مهم

### ۱. متغیرهای محیطی
```bash
# فایل .env برای هر کامپوننت
DATABASE_URL="postgresql://user:pass@localhost:5432/dentalai"
AI_API_URL="http://localhost:8000"
NEXT_PUBLIC_API_URL="http://localhost:7272"
```

### ۲. امنیت
```bash
# تنظیم فایروال
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 22

# تنظیم SSL با Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### ۳. مانیتورینگ
```bash
# نصب monitoring tools
sudo apt install htop iotop ncdu

# PM2 monitoring
pm2 monit
```

---

## 🎯 چک‌لیست Deployment

- [ ] انتقال فایل‌های پروژه
- [ ] راه‌اندازی PostgreSQL
- [ ] تنظیم Python environment و AI server
- [ ] تنظیم Prisma و database migrations
- [ ] Build کردن frontendها
- [ ] تنظیم Nginx reverse proxy
- [ ] تنظیم SSL certificate
- [ ] تست همه endpointها
- [ ] تنظیم backup و monitoring

---

## 🔧 عیب‌یابی

### مشکل اتصال به Database
```bash
# چک کردن اتصال
psql -h localhost -U postgres -d dentalai

# لاگ‌های Prisma
npx prisma studio
```

### مشکل AI Server
```bash
# چک کردن پورت
netstat -tlnp | grep 8000

# لاگ‌های Python
tail -f ~/logs/ai-server.log
```

### مشکل Frontend
```bash
# چک کردن build
npm run build

# چک کردن preview
npm run preview -- --port 3030
```
