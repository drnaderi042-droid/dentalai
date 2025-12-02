# گزارش باگ‌های امنیتی پروژه Dental AI

**تاریخ بررسی:** 2024  
**اولویت:** 🔴 بحرانی | 🟠 بالا | 🟡 متوسط | 🟢 پایین

---

## 🔴 باگ‌های بحرانی (Critical)

### 1. JWT Secret با Fallback ناامن

**موقعیت:** `minimal-api-dev-v6/src/pages/api/auth/sign-in.ts` (خط 11)  
**کد مشکل‌دار:**
```typescript
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
```

**مشکل:**
- اگر متغیر محیطی `JWT_SECRET` تنظیم نشده باشد، از مقدار پیش‌فرض `'your-secret-key'` استفاده می‌شود
- این مقدار قابل حدس زدن است و امنیت JWT را به خطر می‌اندازد
- مهاجم می‌تواند token های جعلی تولید کند

**راه‌حل:**
```typescript
const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  throw new Error('JWT_SECRET environment variable is required');
}
```

**اولویت:** 🔴 بحرانی

---

### 2. CORS با Wildcard (*)

**موقعیت:** `minimal-api-dev-v6/src/pages/api/serve-upload.ts` (خط 8)  
**کد مشکل‌دار:**
```typescript
res.setHeader('Access-Control-Allow-Origin', '*');
```

**مشکل:**
- اجازه دسترسی به API از هر دامنه‌ای داده می‌شود
- این می‌تواند منجر به حملات CSRF شود
- داده‌های حساس در معرض دسترسی غیرمجاز قرار می‌گیرند

**راه‌حل:**
```typescript
const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3030'];
const origin = req.headers.origin;
if (origin && allowedOrigins.includes(origin)) {
  res.setHeader('Access-Control-Allow-Origin', origin);
}
```

**اولویت:** 🔴 بحرانی

---

### 3. XSS Vulnerability با dangerouslySetInnerHTML

**موقعیت:** `vite-js/src/layouts/components/notifications-drawer/notification-item.jsx` (خط 105)  
**کد مشکل‌دار:**
```jsx
<Box
  dangerouslySetInnerHTML={{ __html: data }}
/>
```

**مشکل:**
- محتوای HTML بدون sanitization رندر می‌شود
- مهاجم می‌تواند اسکریپت‌های مخرب را در notification ها تزریق کند
- می‌تواند منجر به سرقت session، دستکاری DOM، یا redirect به سایت‌های مخرب شود

**راه‌حل:**
```jsx
import DOMPurify from 'dompurify';

function reader(data) {
  const sanitizedData = DOMPurify.sanitize(data);
  return (
    <Box
      dangerouslySetInnerHTML={{ __html: sanitizedData }}
    />
  );
}
```

**اولویت:** 🔴 بحرانی

---

## 🟠 باگ‌های با اولویت بالا (High)

### 4. ذخیره‌سازی داده در localStorage

**موقعیت:** 
- `vite-js/src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx`
- `vite-js/src/sections/orthodontics/patient/view/cephalometric-analysis-view.jsx`

**کد مشکل‌دار:**
```javascript
localStorage.setItem(`cephalometric_analysis_confirmed_${id}`, 'true');
localStorage.setItem(`cephalometric_viewing_tables_${id}`, 'true');
```

**مشکل:**
- localStorage در معرض حملات XSS است
- داده‌های حساس نباید در localStorage ذخیره شوند
- اگرچه در این مورد داده‌های حساس نیست، اما الگوی بدی است

**راه‌حل:**
- برای داده‌های حساس از sessionStorage یا state management استفاده کنید
- اگر باید در localStorage ذخیره شود، داده‌ها را encrypt کنید

**اولویت:** 🟠 بالا

---

### 5. Directory Traversal Protection ناکافی

**موقعیت:** `minimal-api-dev-v6/src/pages/api/serve-upload.ts` (خط 29-32)  
**کد مشکل‌دار:**
```typescript
const normalizedPath = path.normalize(filePath);
if (normalizedPath.includes('../') || normalizedPath.includes('..\\')) {
  return res.status(403).json({ message: 'Invalid path' });
}
```

**مشکل:**
- چک کردن فقط برای `../` و `..\\` کافی نیست
- می‌تواند با encoding های مختلف دور زده شود (مثل `%2e%2e%2f`)
- باید از `path.resolve` و مقایسه با base directory استفاده شود

**راه‌حل:**
```typescript
const uploadsDir = path.join(process.cwd(), 'uploads');
const normalizedPath = path.normalize(filePath);
const fullPath = path.resolve(uploadsDir, normalizedPath);

// Ensure the resolved path is within uploads directory
if (!fullPath.startsWith(path.resolve(uploadsDir))) {
  return res.status(403).json({ message: 'Invalid path' });
}
```

**اولویت:** 🟠 بالا

---

### 6. عدم اعتبارسنجی کامل ورودی‌ها

**موقعیت:** `minimal-api-dev-v6/src/pages/api/patients/index.ts`  
**کد مشکل‌دار:**
```typescript
const { firstName, lastName, phone, age, diagnosis, treatment, status, notes, specialty, nextVisitTime, treatmentStartDate } = req.body;
```

**مشکل:**
- ورودی‌ها بدون validation کامل استفاده می‌شوند
- ممکن است مقادیر غیرمنتظره یا مخرب وارد شود
- SQL Injection از طریق Prisma محافظت می‌شود، اما validation لازم است

**راه‌حل:**
```typescript
import { z } from 'zod';

const patientSchema = z.object({
  firstName: z.string().min(1).max(100),
  lastName: z.string().min(1).max(100),
  phone: z.string().regex(/^[0-9+\-() ]+$/),
  age: z.number().int().min(0).max(150),
  diagnosis: z.string().max(1000),
  treatment: z.string().max(2000),
  // ...
});

const validatedData = patientSchema.parse(req.body);
```

**اولویت:** 🟠 بالا

---

### 7. Error Messages اطلاعات حساس لو می‌دهند

**موقعیت:** `minimal-api-dev-v6/src/pages/api/auth/sign-in.ts` (خط 85)  
**کد مشکل‌دار:**
```typescript
console.error('[Auth API]: ', error);
res.status(500).json({
  message: 'Internal server error',
});
```

**مشکل:**
- خطاهای کامل در console لاگ می‌شوند که ممکن است اطلاعات حساس داشته باشند
- در production باید خطاها را sanitize کرد

**راه‌حل:**
```typescript
if (process.env.NODE_ENV === 'production') {
  console.error('[Auth API]: Internal server error');
} else {
  console.error('[Auth API]: ', error);
}
```

**اولویت:** 🟠 بالا

---

## 🟡 باگ‌های با اولویت متوسط (Medium)

### 8. عدم Rate Limiting

**موقعیت:** تمام API endpoints  
**مشکل:**
- هیچ rate limiting برای API endpoints وجود ندارد
- مهاجم می‌تواند حملات brute force یا DDoS انجام دهد

**راه‌حل:**
```typescript
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
```

**اولویت:** 🟡 متوسط

---

### 9. عدم اعتبارسنجی File Upload

**موقعیت:** File upload endpoints  
**مشکل:**
- بررسی کامل نوع فایل و اندازه فایل انجام نمی‌شود
- ممکن است فایل‌های مخرب آپلود شوند

**راه‌حل:**
```typescript
const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];
const maxFileSize = 10 * 1024 * 1024; // 10MB

if (!allowedMimeTypes.includes(file.mimetype)) {
  return res.status(400).json({ message: 'Invalid file type' });
}

if (file.size > maxFileSize) {
  return res.status(400).json({ message: 'File too large' });
}
```

**اولویت:** 🟡 متوسط

---

### 10. عدم استفاده از HTTPS در Production

**موقعیت:** `vite-js/src/utils/axios.js`  
**کد مشکل‌دار:**
```javascript
return `${protocol}//${hostname}:7272`;
```

**مشکل:**
- در production باید از HTTPS استفاده شود
- داده‌ها در transit رمزگذاری نمی‌شوند

**راه‌حل:**
```javascript
const protocol = process.env.NODE_ENV === 'production' ? 'https:' : 'http:';
```

**اولویت:** 🟡 متوسط

---

### 11. Session Management

**موقعیت:** Authentication system  
**مشکل:**
- JWT token ها در localStorage یا memory ذخیره می‌شوند
- هیچ مکانیزم refresh token وجود ندارد
- Token expiration طولانی است (7 روز)

**راه‌حل:**
- استفاده از refresh token با expiration کوتاه‌تر
- ذخیره token در httpOnly cookie برای جلوگیری از XSS
- کاهش expiration time به 15-30 دقیقه

**اولویت:** 🟡 متوسط

---

## 🟢 باگ‌های با اولویت پایین (Low)

### 12. Environment Variables در کد

**موقعیت:** `vite-js/src/config-global.js`  
**مشکل:**
- برخی متغیرهای محیطی ممکن است در کد expose شوند
- باید مطمئن شوید که هیچ secret در کد client-side نیست

**راه‌حل:**
- بررسی کنید که هیچ API key یا secret در کد frontend نیست
- از environment variables برای تمام secrets استفاده کنید

**اولویت:** 🟢 پایین

---

### 13. عدم استفاده از Content Security Policy (CSP)

**موقعیت:** Frontend application  
**مشکل:**
- هیچ CSP header تنظیم نشده است
- می‌تواند از حملات XSS جلوگیری کند

**راه‌حل:**
```html
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';">
```

**اولویت:** 🟢 پایین

---

### 14. عدم Logging و Monitoring

**موقعیت:** تمام سیستم  
**مشکل:**
- لاگ‌های امنیتی کافی وجود ندارد
- هیچ monitoring برای حملات وجود ندارد

**راه‌حل:**
- پیاده‌سازی logging برای:
  - تلاش‌های ناموفق login
  - دسترسی‌های غیرمجاز
  - تغییرات حساس
- استفاده از tools مثل Sentry یا LogRocket

**اولویت:** 🟢 پایین

---

## خلاصه و توصیه‌ها

### اولویت‌بندی رفع باگ‌ها:

1. **فوری (Critical):**
   - رفع JWT Secret fallback
   - محدود کردن CORS
   - رفع XSS vulnerability

2. **در اسرع وقت (High):**
   - بهبود Directory Traversal protection
   - اعتبارسنجی کامل ورودی‌ها
   - بهبود error handling

3. **در آینده نزدیک (Medium):**
   - پیاده‌سازی Rate Limiting
   - بهبود File Upload validation
   - استفاده از HTTPS

4. **بهبودهای امنیتی (Low):**
   - پیاده‌سازی CSP
   - بهبود Logging و Monitoring

---

## منابع و مراجع

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [Node.js Security Best Practices](https://nodejs.org/en/docs/guides/security/)

---

**نکته:** این گزارش بر اساس بررسی کد فعلی تهیه شده است. توصیه می‌شود یک audit امنیتی کامل توسط متخصص امنیت انجام شود.


