# راهنمای کامل Deployment پروژه Dental AI
# Complete Deployment Guide for Dental AI Project

این راهنما شامل تمام مراحل لازم برای آپلود و اجرای پروژه در سرور برای استفاده عمومی است.

---

## 📋 فهرست مطالب (Table of Contents)

1. [پیش‌نیازها (Prerequisites)](#پیش-نیازها)
2. [معماری سیستم (System Architecture)](#معماری-سیستم)
3. [مراحل Deployment](#مراحل-deployment)
4. [تنظیمات Environment Variables](#تنظیمات-environment-variables)
5. [تنظیمات سرور (Server Configuration)](#تنظیمات-سرور)
6. [نکات امنیتی (Security Considerations)](#نکات-امنیتی)
7. [مانیتورینگ و لاگ‌ها (Monitoring & Logging)](#مانیتورینگ-و-لاگها)
8. [Backup و Recovery](#backup-و-recovery)
9. [Troubleshooting](#troubleshooting)

---

## پیش‌نیازها

### سخت‌افزار مورد نیاز (Hardware Requirements)

**حداقل (Minimum):**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- GPU: اختیاری (برای سرعت بیشتر در AI processing)

**توصیه شده (Recommended):**
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 100+ GB SSD
- GPU: NVIDIA GPU با CUDA support (برای AI models)

### نرم‌افزار مورد نیاز (Software Requirements)

1. **Operating System:**
   - Ubuntu 20.04 LTS یا بالاتر (توصیه شده)
   - یا Windows Server 2019+

2. **Node.js:**
   ```bash
   # نصب Node.js 18.x یا بالاتر
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

3. **Python:**
   ```bash
   # نصب Python 3.8 یا بالاتر
   sudo apt-get update
   sudo apt-get install python3.8 python3.8-venv python3-pip
   ```

4. **Database:**
   - PostgreSQL 14+ (توصیه شده)
   - یا MongoDB 5.0+

5. **Web Server:**
   - Nginx (توصیه شده)
   - یا Apache

6. **Process Manager:**
   - PM2 برای Node.js
   - systemd برای Python

---

## معماری سیستم

```
┌─────────────────┐
│   Nginx (80/443) │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐  ┌──▼──────┐
│Frontend│  │Backend │
│(Vite)  │  │(Next.js)│
│:3030   │  │:7272   │
└────────┘  └───┬────┘
                │
         ┌──────▼──────┐
         │Python AI    │
         │Server       │
         │:5001        │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │PostgreSQL/  │
         │MongoDB      │
         └─────────────┘
```

---

## مراحل Deployment

### مرحله 1: آماده‌سازی سرور

```bash
# به‌روزرسانی سیستم
sudo apt-get update && sudo apt-get upgrade -y

# نصب ابزارهای پایه
sudo apt-get install -y git curl wget build-essential

# نصب Nginx
sudo apt-get install -y nginx

# نصب PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# نصب PM2
sudo npm install -g pm2

# نصب Python dependencies
sudo apt-get install -y python3-dev python3-pip python3-venv
```

### مرحله 2: کلون کردن پروژه

```bash
# ایجاد دایرکتوری پروژه
sudo mkdir -p /var/www/dental-ai
sudo chown $USER:$USER /var/www/dental-ai
cd /var/www/dental-ai

# کلون کردن پروژه (یا آپلود فایل‌ها)
# git clone <your-repo-url> .
# یا
# scp -r /path/to/local/project/* user@server:/var/www/dental-ai/
```

### مرحله 3: تنظیم Frontend (Vite-js)

```bash
cd /var/www/dental-ai/vite-js

# نصب dependencies
npm install

# Build برای production
npm run build

# فایل‌های build شده در دایرکتوری dist قرار می‌گیرند
```

**تنظیم Nginx برای Frontend:**

```nginx
# /etc/nginx/sites-available/dental-ai-frontend
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    root /var/www/dental-ai/vite-js/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json;
}
```

```bash
# فعال کردن configuration
sudo ln -s /etc/nginx/sites-available/dental-ai-frontend /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### مرحله 4: تنظیم Backend API (Next.js)

```bash
cd /var/www/dental-ai/minimal-api-dev-v6

# نصب dependencies
npm install

# Build برای production
npm run build

# ایجاد فایل .env.production
cat > .env.production << EOF
NODE_ENV=production
PORT=7272
HOST=0.0.0.0

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dental_ai

# JWT Secret
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production

# AI Server URL
AI_SERVER_URL=http://localhost:5001

# File Upload
MAX_FILE_SIZE=10485760
UPLOAD_DIR=/var/www/dental-ai/uploads
EOF
```

**اجرای Backend با PM2:**

```bash
# ایجاد فایل ecosystem.config.js
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'dental-ai-api',
    script: 'npm',
    args: 'start',
    cwd: '/var/www/dental-ai/minimal-api-dev-v6',
    env: {
      NODE_ENV: 'production',
      PORT: 7272
    },
    error_file: '/var/log/dental-ai/api-error.log',
    out_file: '/var/log/dental-ai/api-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
};
EOF

# اجرای با PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

**تنظیم Nginx برای Backend API:**

```nginx
# /etc/nginx/sites-available/dental-ai-api
server {
    listen 80;
    server_name api.your-domain.com;

    location / {
        proxy_pass http://localhost:7272;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### مرحله 5: تنظیم Python AI Server

```bash
cd /var/www/dental-ai

# ایجاد virtual environment
python3 -m venv .venv
source .venv/bin/activate

# نصب dependencies
pip install --upgrade pip
pip install -r requirements_unified_api.txt

# نصب dependencies اضافی برای face-alignment
pip install face-alignment mediapipe scikit-image

# نصب dependencies برای CLdetection2023 (اگر نیاز باشد)
cd CLdetection2023/mmpose_package/mmpose
pip install -e .
mim install mmengine
mim install "mmcv>=2.0.0"
cd /var/www/dental-ai
```

**ایجاد systemd service برای Python AI Server:**

```bash
sudo cat > /etc/systemd/system/dental-ai-python.service << EOF
[Unit]
Description=Dental AI Python Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/dental-ai
Environment="PATH=/var/www/dental-ai/.venv/bin"
ExecStart=/var/www/dental-ai/.venv/bin/python unified_ai_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# فعال کردن و اجرای service
sudo systemctl daemon-reload
sudo systemctl enable dental-ai-python
sudo systemctl start dental-ai-python
sudo systemctl status dental-ai-python
```

### مرحله 6: تنظیم Database

```bash
# ورود به PostgreSQL
sudo -u postgres psql

# ایجاد database و user
CREATE DATABASE dental_ai;
CREATE USER dental_ai_user WITH PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE dental_ai TO dental_ai_user;
\q

# اجرای migrations (اگر Prisma استفاده می‌شود)
cd /var/www/dental-ai/minimal-api-dev-v6
npx prisma migrate deploy
# یا
npx prisma db push
```

### مرحله 7: تنظیم SSL Certificate (HTTPS)

```bash
# نصب Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# دریافت certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com -d api.your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## تنظیمات Environment Variables

### Frontend (.env.production)

```env
VITE_API_URL=https://api.your-domain.com
VITE_APP_NAME=Dental AI
VITE_APP_VERSION=1.0.0
```

### Backend (.env.production)

```env
NODE_ENV=production
PORT=7272
HOST=0.0.0.0

# Database
DATABASE_URL=postgresql://dental_ai_user:password@localhost:5432/dental_ai

# JWT
JWT_SECRET=your-super-secret-jwt-key-min-32-chars

# AI Server
AI_SERVER_URL=http://localhost:5001

# File Upload
MAX_FILE_SIZE=10485760
UPLOAD_DIR=/var/www/dental-ai/uploads

# CORS
ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com
```

### Python AI Server

```bash
# ایجاد فایل .env در root directory
cat > /var/www/dental-ai/.env << EOF
FLASK_ENV=production
FLASK_DEBUG=False
PORT=5001
HOST=0.0.0.0

# Model Paths
YOLO_MODEL_PATH=/var/www/dental-ai/fyp2.v12i.yolov11_2/weights/best.pt
AARIZ_512_MODEL_PATH=/var/www/dental-ai/Aariz/checkpoints/best_model_512.pth
AARIZ_768_MODEL_PATH=/var/www/dental-ai/Aariz/checkpoints/best_model_768.pth
CLDETECTION_MODEL_PATH=/var/www/dental-ai/CLdetection2023/checkpoints/best.pth

# GPU Settings
CUDA_VISIBLE_DEVICES=0
USE_GPU=True
EOF
```

---

## تنظیمات سرور

### Firewall Configuration

```bash
# فعال کردن UFW
sudo ufw enable

# باز کردن پورت‌های لازم
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp     # HTTP
sudo ufw allow 443/tcp    # HTTPS
sudo ufw allow 7272/tcp   # Backend API (فقط از localhost)
sudo ufw allow 5001/tcp   # Python AI Server (فقط از localhost)

# بررسی وضعیت
sudo ufw status
```

### تنظیمات Nginx

```nginx
# /etc/nginx/nginx.conf
# اضافه کردن در بخش http:

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=general_limit:10m rate=30r/s;

# در server block مربوط به API:
limit_req zone=api_limit burst=20 nodelay;

# در server block مربوط به Frontend:
limit_req zone=general_limit burst=50 nodelay;
```

---

## نکات امنیتی

### 1. تغییر Passwords پیش‌فرض

```bash
# تغییر password برای database user
sudo -u postgres psql
ALTER USER dental_ai_user WITH PASSWORD 'new-secure-password';

# تغییر JWT_SECRET
# در .env.production یک secret key قوی ایجاد کنید
```

### 2. تنظیمات File Permissions

```bash
# تنظیم permissions برای فایل‌ها
sudo chown -R www-data:www-data /var/www/dental-ai
sudo chmod -R 755 /var/www/dental-ai
sudo chmod -R 775 /var/www/dental-ai/uploads
```

### 3. تنظیمات CORS

```javascript
// در minimal-api-dev-v6/src/utils/cors.js
// فقط domain های مجاز را اضافه کنید
const allowedOrigins = [
  'https://your-domain.com',
  'https://www.your-domain.com'
];
```

### 4. Rate Limiting

```bash
# نصب rate limiting middleware در Next.js
npm install express-rate-limit
```

---

## مانیتورینگ و لاگ‌ها

### PM2 Monitoring

```bash
# مشاهده وضعیت
pm2 status
pm2 logs dental-ai-api

# Monitoring dashboard
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
```

### Systemd Logs

```bash
# مشاهده لاگ‌های Python AI Server
sudo journalctl -u dental-ai-python -f

# مشاهده لاگ‌های Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Health Checks

```bash
# ایجاد endpoint برای health check
# در unified_ai_api_server.py:
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200
```

---

## Backup و Recovery

### Database Backup

```bash
# ایجاد script برای backup
cat > /usr/local/bin/backup-dental-ai-db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/dental-ai"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
pg_dump -U dental_ai_user dental_ai > $BACKUP_DIR/db_backup_$DATE.sql
# حذف backup های قدیمی‌تر از 7 روز
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
EOF

chmod +x /usr/local/bin/backup-dental-ai-db.sh

# اضافه کردن به crontab (هر روز ساعت 2 صبح)
crontab -e
# اضافه کردن:
0 2 * * * /usr/local/bin/backup-dental-ai-db.sh
```

### File Backup

```bash
# Backup فایل‌های upload شده
tar -czf /var/backups/dental-ai/uploads_$(date +%Y%m%d).tar.gz /var/www/dental-ai/uploads
```

---

## Troubleshooting

### مشکل: Frontend نمایش داده نمی‌شود

```bash
# بررسی Nginx
sudo nginx -t
sudo systemctl status nginx

# بررسی فایل‌های build
ls -la /var/www/dental-ai/vite-js/dist

# بررسی لاگ‌ها
sudo tail -f /var/log/nginx/error.log
```

### مشکل: Backend API کار نمی‌کند

```bash
# بررسی PM2
pm2 status
pm2 logs dental-ai-api

# بررسی پورت
sudo netstat -tlnp | grep 7272

# بررسی database connection
cd /var/www/dental-ai/minimal-api-dev-v6
npx prisma db pull
```

### مشکل: Python AI Server کار نمی‌کند

```bash
# بررسی systemd service
sudo systemctl status dental-ai-python
sudo journalctl -u dental-ai-python -n 50

# بررسی virtual environment
source /var/www/dental-ai/.venv/bin/activate
python -c "import flask; print('Flask OK')"

# بررسی پورت
sudo netstat -tlnp | grep 5001
```

### مشکل: Database Connection Error

```bash
# بررسی PostgreSQL
sudo systemctl status postgresql

# بررسی connection
sudo -u postgres psql -c "SELECT version();"

# بررسی user و database
sudo -u postgres psql -c "\du"
sudo -u postgres psql -c "\l"
```

---

## مراحل نهایی

1. **تست تمام Endpoints:**
   ```bash
   # Frontend
   curl http://your-domain.com
   
   # Backend API
   curl http://api.your-domain.com/api/health
   
   # Python AI Server
   curl http://localhost:5001/health
   ```

2. **بررسی Performance:**
   - استفاده از Google PageSpeed Insights
   - استفاده از GTmetrix
   - بررسی response times

3. **تنظیمات Monitoring:**
   - نصب monitoring tools (مثل New Relic, Datadog)
   - تنظیم alerts برای errors و downtime

4. **Documentation:**
   - ایجاد API documentation
   - ایجاد user guide

---

## نکات مهم

1. **همیشه از HTTPS استفاده کنید** - برای امنیت داده‌ها
2. **Backup منظم** - حداقل روزانه
3. **Update منظم** - به‌روزرسانی dependencies و security patches
4. **Monitoring** - نظارت مداوم بر performance و errors
5. **Logging** - نگهداری لاگ‌ها برای troubleshooting

---

## پشتیبانی

در صورت بروز مشکل، لاگ‌ها را بررسی کنید و از documentation استفاده کنید.

**لاگ‌های مهم:**
- `/var/log/nginx/error.log` - Nginx errors
- `pm2 logs dental-ai-api` - Backend API logs
- `sudo journalctl -u dental-ai-python` - Python AI Server logs
- `/var/log/postgresql/` - Database logs

---

**موفق باشید! 🚀**

