# راهنمای راه‌اندازی Backend API

## ✅ مراحل اجرای Backend

### ۱. بررسی Dependencies

```bash
cd minimal-api-dev-v6
npm list uuid multer
```

اگر پیام `(empty)` دیدید، dependencies را نصب کنید:

```bash
npm install
```

### ۲. اطمینان از آزاد بودن Port 7272

#### چک کردن:
```powershell
Get-NetTCPConnection -LocalPort 7272
```

#### اگر پورت پر است، آزاد کنید:
```powershell
Get-NetTCPConnection -LocalPort 7272 | ForEach-Object { 
    Stop-Process -Id $_.OwningProcess -Force 
}
```

### ۳. اجرای Backend

```bash
cd minimal-api-dev-v6
npm run dev
```

Backend باید روی **http://localhost:7272** اجرا شود.

### ۴. تست Backend

#### تست ساده (PowerShell):
```powershell
# Test if backend is responding
Invoke-WebRequest -Uri "http://localhost:7272/api/chat/doctors" -UseBasicParsing
```

#### تست با curl:
```bash
curl http://localhost:7272/api/chat/doctors
```

---

## 🔍 مشکلات رایج

### Problem 1: "Module not found: Can't resolve 'uuid'"
**علت:** Dependencies نصب نشده‌اند.  
**راه‌حل:**
```bash
cd minimal-api-dev-v6
npm install
```

### Problem 2: "EADDRINUSE: address already in use :::7272"
**علت:** یک process دیگر روی پورت 7272 اجرا می‌شود.  
**راه‌حل:**
```powershell
# پیدا کردن و stop کردن process:
Get-NetTCPConnection -LocalPort 7272 | ForEach-Object { 
    $proc = Get-Process -Id $_.OwningProcess
    Write-Host "Stopping $($proc.Name) (PID: $($proc.Id))"
    Stop-Process -Id $proc.Id -Force
}
```

### Problem 3: "The operation has timed out"
**علت:** Backend در حال compile است (اولین بار کمی طول می‌کشد).  
**راه‌حل:** صبر کنید تا Next.js compile شود (تا 30 ثانیه).

---

## 📊 بررسی وضعیت Backend

### چک کردن اینکه Backend در حال اجرا است:

```powershell
$port = Get-NetTCPConnection -LocalPort 7272 -ErrorAction SilentlyContinue
if ($port) {
    $proc = Get-Process -Id $port.OwningProcess
    Write-Host "✅ Backend is running: $($proc.Name) (PID: $($proc.Id))" -ForegroundColor Green
} else {
    Write-Host "❌ Backend is NOT running" -ForegroundColor Red
}
```

### تست API Endpoints:

```bash
# Test doctors list (should return 401 without auth, which means it's working)
curl http://localhost:7272/api/chat/doctors

# Test upload endpoint (should return 401 or 405)
curl -X OPTIONS http://localhost:7272/api/upload/chat
```

---

## 🎯 Backend فعال است - حالا چه کنیم؟

1. **Frontend (vite-js) را اجرا کنید:**
   ```bash
   cd vite-js
   npm run dev
   ```

2. **به http://localhost:5173 بروید**

3. **لاگین کنید با یک حساب دکتر**

4. **به صفحه Chat بروید (/dashboard/chat)**

5. **ارسال عکس را تست کنید! ✅**

---

## 🚨 در صورت مشکل:

1. لاگ‌های Backend را چک کنید (در Terminal که backend اجرا کردید)
2. Browser Console را چک کنید (F12)
3. مطمئن شوید که لاگین کرده‌اید و role شما DOCTOR است

---

## 📝 نکات مهم:

- ✅ Backend بر پایه **Next.js** است (برای API Routes)
- ✅ Frontend بر پایه **Vite + React** است
- ✅ Backend باید **همیشه** در حال اجرا باشد تا Frontend کار کند
- ✅ هر دو باید همزمان اجرا شوند:
  - Backend: `http://localhost:7272`
  - Frontend: `http://localhost:5173`

---

## ✅ چک لیست نهایی:

- [ ] `npm install` در `minimal-api-dev-v6` اجرا شد
- [ ] Port 7272 آزاد است
- [ ] Backend اجرا شده (`npm run dev` در `minimal-api-dev-v6`)
- [ ] Frontend اجرا شده (`npm run dev` در `vite-js`)
- [ ] لاگین کرده‌اید
- [ ] Role شما DOCTOR است

اگر همه این‌ها چک شدند، چت باید کاملاً کار کند! 🎉

