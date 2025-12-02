# مثال سریع استفاده از Performance Monitor

## 🚀 استفاده سریع در صفحه Patient

صفحه `dashboard/orthodontics/patient/[id]` قبلاً تنظیم شده است. فقط کافیست:

1. سرور development را اجرا کنید:
```bash
npm run dev
# یا
yarn dev
```

2. به صفحه بروید:
```
http://localhost:3030/dashboard/orthodontics/patient/cmhqv0h4w0011amdb4txuy7z0
```

3. در گوشه پایین راست صفحه، یک کارت Performance Monitor خواهید دید.

4. روی کارت کلیک کنید تا باز شود و متریک‌ها را ببینید.

---

## 📊 چه اطلاعاتی نمایش داده می‌شود؟

### حافظه (RAM)
- **استفاده شده**: مقدار حافظه استفاده شده (MB)
- **کل**: کل حافظه تخصیص یافته (MB)
- **درصد**: درصد استفاده از حافظه
- **رنگ**: سبز (<50%), زرد (50-80%), قرمز (>80%)

### پردازنده (CPU)
- **استفاده**: درصد استفاده از CPU
- **بار پردازشی**: بار کلی سیستم
- **رنگ**: سبز (<30%), زرد (30-70%), قرمز (>70%)

### رندر
- **زمان رندر**: زمان آخرین render (میلی‌ثانیه)
- **تعداد رندر**: تعداد دفعاتی که کامپوننت render شده

---

## 🔧 استفاده در کامپوننت‌های دیگر

### روش 1: استفاده مستقیم

```jsx
import { PerformanceMonitor } from 'src/components/performance-monitor';

function MyComponent() {
  return (
    <>
      <div>محتوای کامپوننت</div>
      
      {import.meta.env.DEV && (
        <PerformanceMonitor 
          componentName="MyComponent" 
          position="bottom-right"
        />
      )}
    </>
  );
}
```

### روش 2: استفاده با Hook

```jsx
import { usePerformanceMonitor } from 'src/hooks/use-performance-monitor';

function MyComponent() {
  const metrics = usePerformanceMonitor('MyComponent');
  
  // استفاده از metrics
  console.log('Memory:', metrics.memory.percentage);
  console.log('CPU:', metrics.cpu.usage);
  
  return <div>محتوای کامپوننت</div>;
}
```

### روش 3: ردیابی چند کامپوننت

```jsx
import { PerformanceDashboard } from 'src/components/performance-monitor';

function MyPage() {
  return (
    <>
      <Header />
      <Sidebar />
      <MainContent />
      
      {import.meta.env.DEV && (
        <PerformanceDashboard 
          components={['Header', 'Sidebar', 'MainContent']}
          position="bottom-right"
        />
      )}
    </>
  );
}
```

---

## ⚙️ تنظیمات

### موقعیت مانیتور
- `top-left`: بالا چپ
- `top-right`: بالا راست
- `bottom-left`: پایین چپ
- `bottom-right`: پایین راست (پیش‌فرض)

### فاصله به‌روزرسانی
```jsx
<PerformanceMonitor 
  componentName="MyComponent"
  interval={2000}  // هر 2 ثانیه به‌روزرسانی
/>
```

### نمایش خودکار
```jsx
<PerformanceMonitor 
  componentName="MyComponent"
  showOnMount={true}  // نمایش خودکار هنگام mount
/>
```

---

## 🎯 نکات مهم

1. **فقط در Development**: مانیتورها فقط در حالت development نمایش داده می‌شوند
2. **Browser Support**: ردیابی حافظه در Chrome/Edge بهتر کار می‌کند
3. **Performance Impact**: استفاده از مانیتورها خودش کمی overhead دارد
4. **CPU Estimation**: ردیابی CPU در مرورگر تخمینی است

---

## 🐛 عیب‌یابی

### مانیتور نمایش داده نمی‌شود؟
- مطمئن شوید که `import.meta.env.DEV` برابر `true` است
- بررسی کنید که کامپوننت mount شده است

### متریک‌ها به‌روز نمی‌شوند؟
- بررسی کنید که `interval` مقدار مناسبی دارد
- در Chrome DevTools بررسی کنید

---

## 📚 مستندات کامل

برای اطلاعات بیشتر، فایل `PERFORMANCE_MONITORING_GUIDE.md` را مطالعه کنید.


