# 🚀 راهنمای سریع Deployment

این فایل راهنمای سریع برای deployment پروژه Dental AI است. برای راهنمای کامل، به `DEPLOYMENT_GUIDE.md` مراجعه کنید.

## ⚡ شروع سریع (Quick Start)

### استفاده از اسکریپت خودکار:

```bash
# Clone یا آپلود پروژه به سرور
cd /var/www
sudo git clone <your-repo-url> dental-ai
# یا
sudo scp -r /path/to/local/project/* user@server:/var/www/dental-ai/

# اجرای اسکریپت deployment
cd dental-ai
sudo chmod +x deploy.sh
sudo ./deploy.sh your-domain.com

# پس از اجرای اسکریپت، مراحل زیر را انجام دهید:
```

### مراحل دستی:

```bash
# 1. نصب dependencies
cd /var/www/dental-ai
source .venv/bin/activate
pip install -r requirements_unified_api.txt
pip install face-alignment mediapipe scikit-image

# 2. Build Frontend
cd vite-js
npm install
npm run build

# 3. Build Backend
cd ../minimal-api-dev-v6
npm install
npm run build
npm start  # یا با PM2: pm2 start ecosystem.config.js

# 4. اجرای Python AI Server
cd ..
sudo systemctl start dental-ai-python

# 5. تنظیم SSL
sudo certbot --nginx -d your-domain.com -d www.your-domain.com -d api.your-domain.com
```

## 📁 ساختار پروژه

```
/var/www/dental-ai/
├── vite-js/              # Frontend (React + Vite)
│   └── dist/             # Build output
├── minimal-api-dev-v6/   # Backend API (Next.js)
│   └── .next/            # Build output
├── unified_ai_api_server.py  # Python AI Server
├── .venv/                # Python virtual environment
├── uploads/              # Uploaded files
└── ...
```

## 🔧 پورت‌ها

- **Frontend**: 80, 443 (Nginx)
- **Backend API**: 7272 (PM2)
- **Python AI Server**: 5001 (systemd)

## 🔐 Environment Variables

### Backend (.env.production):
```env
NODE_ENV=production
PORT=7272
DATABASE_URL=postgresql://user:pass@localhost:5432/dental_ai
JWT_SECRET=your-secret-key
AI_SERVER_URL=http://localhost:5001
```

### Frontend (.env.production):
```env
VITE_API_URL=https://api.your-domain.com
```

## 📊 بررسی وضعیت

```bash
# بررسی PM2
pm2 status
pm2 logs dental-ai-api

# بررسی Python AI Server
sudo systemctl status dental-ai-python
sudo journalctl -u dental-ai-python -f

# بررسی Nginx
sudo systemctl status nginx
sudo nginx -t

# بررسی Database
sudo systemctl status postgresql
sudo -u postgres psql -c "\l"
```

## 🔄 Restart Services

```bash
# Restart Backend
pm2 restart dental-ai-api

# Restart Python AI Server
sudo systemctl restart dental-ai-python

# Restart Nginx
sudo systemctl restart nginx
```

## 📝 لاگ‌ها

- **Backend**: `pm2 logs dental-ai-api`
- **Python AI Server**: `sudo journalctl -u dental-ai-python -f`
- **Nginx**: `/var/log/nginx/error.log`

## 🆘 مشکلات رایج

### Frontend نمایش داده نمی‌شود
```bash
sudo nginx -t
sudo systemctl restart nginx
ls -la /var/www/dental-ai/vite-js/dist
```

### Backend کار نمی‌کند
```bash
pm2 logs dental-ai-api
pm2 restart dental-ai-api
```

### Python AI Server کار نمی‌کند
```bash
sudo systemctl status dental-ai-python
sudo journalctl -u dental-ai-python -n 50
```

## 📚 مستندات بیشتر

- **راهنمای کامل**: `DEPLOYMENT_GUIDE.md`
- **چک‌لیست**: `QUICK_DEPLOYMENT_CHECKLIST.md`
- **اسکریپت خودکار**: `deploy.sh`

---

**برای اطلاعات بیشتر، به `DEPLOYMENT_GUIDE.md` مراجعه کنید.**

