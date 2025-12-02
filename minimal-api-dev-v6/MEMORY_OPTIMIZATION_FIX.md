# بهینه‌سازی مصرف حافظه - Memory Optimization Fix

## مشکلات شناسایی شده

پروژه Node.js حدود 3-4 گیگابایت RAM مصرف می‌کرد که غیرطبیعی بود. مشکلات اصلی:

### 1. ایجاد چندین نمونه PrismaClient
- هر فایل API یک نمونه جدید `PrismaClient` ایجاد می‌کرد
- هر request باعث ایجاد و قطع اتصال می‌شد
- منجر به نشت حافظه و مصرف بیش از حد RAM

### 2. بارگذاری تمام تصاویر بدون Pagination
- API `/api/patients` تمام تصاویر رادیولوژی را یکجا بارگذاری می‌کرد
- با تعداد زیاد بیماران و تصاویر، حافظه به شدت مصرف می‌شد

### 3. ذخیره فایل‌های آپلود شده در حافظه
- Multer با `memoryStorage()` فایل‌ها را در RAM نگه می‌داشت
- فایل‌های بزرگ (تا 10MB) باعث مصرف شدید RAM می‌شدند

### 4. تبدیل همزمان همه تصاویر به Base64
- API AI همه تصاویر را همزمان به base64 تبدیل می‌کرد
- برای تصاویر بزرگ این کار حافظه زیادی مصرف می‌کرد

## راه‌حل‌های پیاده‌سازی شده

### ✅ 1. ایجاد PrismaClient Singleton
**فایل:** `src/lib/prisma.ts`
- یک نمونه واحد PrismaClient برای کل پروژه
- استفاده مجدد از connection pool
- جلوگیری از قطع و وصل مکرر اتصالات

**تغییرات:**
```typescript
// قبل:
const prisma = new PrismaClient();
await prisma.$disconnect(); // در هر request

// بعد:
import { prisma } from 'src/lib/prisma';
// بدون disconnect - استفاده از connection pool
```

### ✅ 2. افزودن Pagination و حذف include تصاویر
**فایل:** `src/pages/api/patients/index.ts` و `src/pages/api/patients/[id].ts`

**تغییرات:**
- استفاده از `select` به جای `include` برای کنترل دقیق فیلدها
- اضافه کردن pagination (20 مورد در هر صفحه)
- نمایش فقط تعداد تصاویر (`_count`) به جای بارگذاری کامل
- کاهش چشمگیر مصرف حافظه با عدم بارگذاری تصاویر

**قبل:**
```typescript
include: { radiologyImages: true } // بارگذاری همه تصاویر
```

**بعد:**
```typescript
select: {
  // فقط فیلدهای مورد نیاز
  _count: { select: { radiologyImages: true } } // فقط تعداد
}
pagination: { page, limit, total }
```

### ✅ 3. تغییر Multer از Memory به Disk Storage
**فایل:** `src/pages/api/patients/[id]/images.ts`

**تغییرات:**
- تغییر از `multer.memoryStorage()` به `multer.diskStorage()`
- فایل‌ها مستقیماً روی دیسک ذخیره می‌شوند
- عدم مصرف RAM برای فایل‌های آپلود شده

**قبل:**
```typescript
storage: multer.memoryStorage() // فایل در RAM
await fs.writeFile(filePath, file.buffer); // کپی به دیسک
```

**بعد:**
```typescript
storage: multer.diskStorage({ ... }) // مستقیماً روی دیسک
// فایل از قبل روی دیسک است
```

### ✅ 4. محدود کردن و Sequential Processing تصاویر
**فایل:** `src/pages/api/ai/dental-diagnosis.ts`

**تغییرات:**
- محدود کردن تعداد تصاویر به 5 مورد (قبلاً 10)
- پردازش Sequential به جای Parallel
- امکان Garbage Collection بین پردازش تصاویر

**قبل:**
```typescript
const base64Images = await Promise.all(
  images.slice(0, 10).map(url => imageUrlToBase64(url))
); // همه همزمان
```

**بعد:**
```typescript
const maxImages = Math.min(images.length, 5); // محدود به 5
for (let i = 0; i < maxImages; i++) {
  const base64 = await imageUrlToBase64(images[i]); // Sequential
  if (global.gc) global.gc(); // GC بین تصاویر
}
```

### ✅ 5. Pagination برای API Images
**فایل:** `src/pages/api/patients/[id]/images.ts`

**تغییرات:**
- اضافه کردن pagination برای لیست تصاویر
- 20 تصویر در هر صفحه به صورت پیش‌فرض
- فقط metadata بارگذاری می‌شود، نه محتوای فایل

### ✅ 6. تنظیمات Next.js برای بهینه‌سازی حافظه
**فایل:** `next.config.mjs`

**تغییرات:**
- محدود کردن اندازه request body
- غیرفعال کردن minification در development
- کاهش مصرف حافظه در build process

## نتایج مورد انتظار

پس از این بهینه‌سازی‌ها:

1. **کاهش مصرف RAM:** از 3-4GB به حدود 500MB-1GB
2. **بهبود Performance:** سریع‌تر شدن response time
3. **پایداری بیشتر:** کاهش احتمال crash به دلیل کمبود حافظه
4. **Scalability بهتر:** امکان مدیریت تعداد بیشتری از درخواست‌ها

## نحوه استفاده

### اجرای پروژه
```bash
cd minimal-api-dev-v6
npm run dev
```

### بررسی مصرف حافظه
```bash
# Windows PowerShell
Get-Process node | Select-Object ProcessName, @{Name="Memory(MB)";Expression={[math]::Round($_.WS/1MB,2)}}
```

## نکات مهم

1. **PrismaClient:** دیگر نیازی به `$disconnect()` در هر request نیست
2. **Pagination:** Frontend باید از پارامترهای `page` و `limit` استفاده کند
3. **Images API:** Response شامل `pagination` object است که باید استفاده شود
4. **Memory Monitoring:** مصرف حافظه را به طور منظم بررسی کنید

## فایل‌های تغییر یافته

- ✅ `src/lib/prisma.ts` (جدید)
- ✅ `src/pages/api/patients/index.ts`
- ✅ `src/pages/api/patients/[id].ts`
- ✅ `src/pages/api/patients/[id]/images.ts`
- ✅ `src/pages/api/ai/dental-diagnosis.ts`
- ✅ `next.config.mjs`

## فایل‌های باقیمانده

برخی فایل‌های دیگر هنوز از `new PrismaClient()` استفاده می‌کنند. برای بهینه‌سازی کامل، این فایل‌ها نیز باید به singleton تبدیل شوند:

- `src/pages/api/invoice/*.ts`
- `src/pages/api/exchange-rate/index.ts`
- `src/pages/api/patients/[id]/facial-landmark-analysis.ts`
- `src/pages/api/patients/[id]/intraoral-analysis.ts`
- `src/pages/api/auth/*.ts`

---

**تاریخ:** 2025-01-08
**نسخه:** 1.0








