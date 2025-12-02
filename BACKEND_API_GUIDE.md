# 🖥️ راهنمای راه‌اندازی Backend API

## مشکل

```
POST http://localhost:7272/api/ai-model-tests 500 (Internal Server Error)
```

این خطا به این معنی است که **Backend API روی پورت 7272 در حال اجرا نیست** یا مشکل دارد.

---

## 📊 سرویس‌های Backend

پروژه شما **2 سرویس Backend** دارد:

### 1️⃣ HRNet Service (پورت 5000) ✅ در حال اجرا

```
Location: cephx_service/
Purpose: تشخیص Landmarks با مدل HRNet
Port: 5000
Status: ✅ RUNNING
```

**چک کردن:**
```bash
curl http://localhost:5000/health
```

**پاسخ باید:**
```json
{
  "status": "healthy",
  "model": "HRNet-W32",
  "landmarks": 19
}
```

---

### 2️⃣ Next.js API (پورت 7272) ❌ در حال اجرا نیست

```
Location: minimal-api-dev-v6/
Purpose: مدیریت Database (تاریخچه تست‌ها، بیماران، و...)
Port: 7272
Status: ❌ NOT RUNNING
```

**این سرویس برای:**
- ذخیره تاریخچه تست‌های AI
- مدیریت بیماران
- ذخیره داده‌های Cephalometric
- و سایر APIهای CRUD

---

## 🚀 راه‌اندازی Backend API

### گام 1: رفتن به پوشه Backend

```bash
cd "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\minimal-api-dev-v6"
```

### گام 2: نصب Dependencies (اگر نصب نشده)

```bash
npm install
```

### گام 3: Setup Database (Prisma)

```bash
# Generate Prisma Client
npx prisma generate

# اگر migration نداشتید، اجرا کنید:
npx prisma migrate dev
```

### گام 4: اجرای Server

```bash
npm run dev
```

**یا:**

```bash
# برای Windows PowerShell
$env:PORT=7272
npm run dev
```

**خروجی باید:**
```
✓ Ready in X.XXs
○ Local:    http://localhost:7272
```

---

## 🧪 تست API

### چک کردن Health:

```bash
# Test با curl
curl http://localhost:7272/api/health

# یا در مرورگر:
http://localhost:7272/api/health
```

### تست ذخیره Test:

```bash
curl -X POST http://localhost:7272/api/ai-model-tests \
  -H "Content-Type: application/json" \
  -d '{
    "modelId": "local/hrnet-w32",
    "modelName": "HRNet-W32",
    "modelProvider": "Local Server",
    "success": true,
    "processingTime": 1.5
  }'
```

**پاسخ باید:**
```json
{
  "success": true,
  "data": {
    "id": "...",
    "modelId": "local/hrnet-w32",
    ...
  }
}
```

---

## 🔧 حل مشکلات رایج

### مشکل 1: `PrismaClient` پیدا نمی‌شود

```
Error: @prisma/client did not initialize yet
```

**راه حل:**
```bash
cd minimal-api-dev-v6
npx prisma generate
```

---

### مشکل 2: Database نمی‌تواند پیدا شود

```
Error: P1003: Database does not exist
```

**راه حل:**
```bash
npx prisma migrate dev --name init
```

---

### مشکل 3: پورت 7272 در استفاده است

```
Error: Port 7272 is already in use
```

**راه حل 1 - تغییر پورت:**
```bash
$env:PORT=7273
npm run dev
```

سپس در `vite-js/src/config-global.js` پورت را تغییر دهید:
```javascript
export const CONFIG = {
  site: {
    serverUrl: 'http://localhost:7273',
  },
};
```

**راه حل 2 - بستن پروسه قبلی:**
```bash
# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 7272).OwningProcess | Stop-Process -Force
```

---

### مشکل 4: CORS Error

اگر خطای CORS دریافت کردید:

**چک کنید که `index.ts` شامل CORS headers باشد:**

```typescript
res.setHeader('Access-Control-Allow-Origin', '*');
res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
```

✅ این هدرها در API موجود است.

---

## 📋 Checklist برای راه‌اندازی کامل

- [ ] HRNet Service روی پورت 5000 در حال اجرا است
- [ ] Next.js API روی پورت 7272 در حال اجرا است
- [ ] Prisma Client generate شده است
- [ ] Database migration اجرا شده است
- [ ] Frontend روی پورت 3030 در حال اجرا است

---

## 🎯 دستورات کامل (یکجا)

### Terminal 1: HRNet Service

```powershell
cd "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\cephx_service"
.\venv\Scripts\python.exe app_hrnet_real.py
```

### Terminal 2: Next.js API

```powershell
cd "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\minimal-api-dev-v6"
npx prisma generate
npm run dev
```

### Terminal 3: Frontend (Vite)

```powershell
cd "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\vite-js"
npm run dev
```

---

## ✅ بعد از راه‌اندازی

شما باید 3 سرویس در حال اجرا داشته باشید:

| سرویس | پورت | URL | وضعیت |
|-------|------|-----|--------|
| **HRNet AI Model** | 5000 | http://localhost:5000 | ✅ |
| **Next.js Backend API** | 7272 | http://localhost:7272 | ✅ |
| **Vite Frontend** | 3030 | http://localhost:3030 | ✅ |

---

## 🧪 تست نهایی

1. مرورگر را باز کنید: http://localhost:3030
2. به `/dashboard/ai-model-test` بروید
3. یک تصویر Cephalometric آپلود کنید
4. مدل HRNet را انتخاب کنید
5. "شروع تست" را بزنید
6. باید Landmarks تشخیص داده شود ✅
7. در تاریخچه تست ذخیره شود ✅

اگر همه مراحل کار کرد، **همه چیز آماده است!** 🎉

---

تاریخ: 30 اکتبر 2025

