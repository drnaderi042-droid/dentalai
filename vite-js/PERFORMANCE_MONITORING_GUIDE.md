# راهنمای استفاده از Performance Monitor

این راهنما نحوه استفاده از ابزار نظارت بر عملکرد (CPU و RAM) برای کامپوننت‌های React را توضیح می‌دهد.

## 📋 فهرست مطالب

1. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
2. [استفاده پایه](#استفاده-پایه)
3. [استفاده پیشرفته](#استفاده-پیشرفته)
4. [مثال‌های کاربردی](#مثال‌های-کاربردی)

---

## نصب و راه‌اندازی

تمام فایل‌های لازم در پروژه موجود است. نیازی به نصب پکیج اضافی نیست.

### فایل‌های ایجاد شده:

- `src/hooks/use-performance-monitor.js` - هوک برای ردیابی عملکرد
- `src/components/performance-monitor/performance-monitor.jsx` - کامپوننت نمایش متریک‌ها
- `src/components/performance-monitor/performance-dashboard.jsx` - داشبورد برای چند کامپوننت
- `src/components/performance-monitor/with-performance-monitor.jsx` - HOC برای wrap کردن کامپوننت‌ها

---

## استفاده پایه

### 1. استفاده مستقیم در صفحه

```jsx
import { PerformanceMonitor } from 'src/components/performance-monitor';

export default function MyPage() {
  return (
    <>
      <MyComponent />
      
      {/* نمایش مانیتور عملکرد */}
      {import.meta.env.DEV && (
        <PerformanceMonitor 
          componentName="MyComponent" 
          position="bottom-right"
          showOnMount={false}
        />
      )}
    </>
  );
}
```

### 2. استفاده با Hook

```jsx
import { usePerformanceMonitor } from 'src/hooks/use-performance-monitor';

function MyComponent() {
  const metrics = usePerformanceMonitor('MyComponent', {
    interval: 1000, // به‌روزرسانی هر 1 ثانیه
    trackMemory: true,
    trackCPU: true,
  });

  // استفاده از metrics در کامپوننت
  console.log('Memory:', metrics.memory);
  console.log('CPU:', metrics.cpu);
  console.log('Render Time:', metrics.renderTime);

  return <div>...</div>;
}
```

### 3. استفاده با HOC

```jsx
import { withPerformanceMonitor } from 'src/components/performance-monitor';

function MyComponent() {
  return <div>...</div>;
}

// Wrap کردن کامپوننت
export default withPerformanceMonitor(MyComponent, {
  componentName: 'MyComponent',
  showMonitor: true,
  position: 'bottom-right',
});
```

---

## استفاده پیشرفته

### 1. ردیابی چندین کامپوننت به صورت همزمان

```jsx
import { PerformanceDashboard } from 'src/components/performance-monitor';

export default function MyPage() {
  return (
    <>
      <Component1 />
      <Component2 />
      <Component3 />
      
      {/* داشبورد عملکرد برای همه کامپوننت‌ها */}
      {import.meta.env.DEV && (
        <PerformanceDashboard 
          components={['Component1', 'Component2', 'Component3']}
          position="bottom-right"
        />
      )}
    </>
  );
}
```

### 2. تنظیمات پیشرفته Hook

```jsx
const metrics = usePerformanceMonitor('MyComponent', {
  interval: 500,        // به‌روزرسانی هر 500ms
  trackMemory: true,    // ردیابی حافظه
  trackCPU: true,       // ردیابی CPU
});
```

### 3. موقعیت‌های مختلف مانیتور

```jsx
<PerformanceMonitor 
  componentName="MyComponent"
  position="top-left"      // یا 'top-right', 'bottom-left', 'bottom-right'
  showOnMount={true}       // نمایش خودکار هنگام mount
/>
```

---

## مثال‌های کاربردی

### مثال 1: صفحه Patient Orthodontics

```jsx
// src/pages/dashboard/orthodontics/patient/[id].jsx
import { PerformanceMonitor } from 'src/components/performance-monitor';

export default function Page() {
  return (
    <>
      <PatientOrthodonticsView />
      
      {import.meta.env.DEV && (
        <PerformanceMonitor 
          componentName="PatientOrthodonticsView" 
          position="bottom-right"
        />
      )}
    </>
  );
}
```

### مثال 2: ردیابی کامپوننت‌های داخلی

```jsx
import { usePerformanceMonitor } from 'src/hooks/use-performance-monitor';

function HeavyComponent() {
  const metrics = usePerformanceMonitor('HeavyComponent');

  // نمایش هشدار در صورت مصرف زیاد
  useEffect(() => {
    if (metrics.memory.percentage > 80) {
      console.warn('⚠️ مصرف حافظه بالا:', metrics.memory.percentage);
    }
    if (metrics.cpu.usage > 70) {
      console.warn('⚠️ مصرف CPU بالا:', metrics.cpu.usage);
    }
  }, [metrics]);

  return <div>...</div>;
}
```

### مثال 3: ردیابی عملکرد در کامپوننت‌های مختلف

```jsx
import { PerformanceDashboard } from 'src/components/performance-monitor';

function ComplexPage() {
  return (
    <>
      <Header />
      <Sidebar />
      <MainContent />
      <Footer />
      
      {import.meta.env.DEV && (
        <PerformanceDashboard 
          components={['Header', 'Sidebar', 'MainContent', 'Footer']}
          position="bottom-right"
        />
      )}
    </>
  );
}
```

---

## 📊 متریک‌های قابل ردیابی

### Memory (RAM)
- **used**: حافظه استفاده شده (MB)
- **total**: کل حافظه تخصیص یافته (MB)
- **limit**: محدودیت حافظه (MB)
- **percentage**: درصد استفاده از حافظه

### CPU
- **usage**: درصد استفاده از CPU
- **load**: بار پردازشی

### Render
- **renderTime**: زمان رندر آخرین render (ms)
- **renderCount**: تعداد رندرها

---

## ⚙️ تنظیمات

### PerformanceMonitor Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `componentName` | string | required | نام کامپوننت برای شناسایی |
| `interval` | number | 1000 | فاصله به‌روزرسانی (ms) |
| `position` | string | 'bottom-right' | موقعیت مانیتور |
| `showOnMount` | boolean | false | نمایش خودکار هنگام mount |

### usePerformanceMonitor Options

| Option | Type | Default | Description |
|-------|------|---------|-------------|
| `interval` | number | 1000 | فاصله به‌روزرسانی (ms) |
| `trackMemory` | boolean | true | ردیابی حافظه |
| `trackCPU` | boolean | true | ردیابی CPU |

---

## 🔍 نکات مهم

1. **فقط در Development**: مانیتورها فقط در حالت development نمایش داده می‌شوند
2. **Performance Impact**: استفاده از مانیتورها خودش کمی overhead دارد، پس فقط در development استفاده کنید
3. **Browser Support**: برای ردیابی حافظه، مرورگر باید از `performance.memory` پشتیبانی کند (Chrome/Edge)
4. **CPU Estimation**: ردیابی CPU در مرورگر تخمینی است و دقیق نیست

---

## 🐛 عیب‌یابی

### مشکل: مانیتور نمایش داده نمی‌شود

- مطمئن شوید که `import.meta.env.DEV` برابر `true` است
- بررسی کنید که کامپوننت mount شده است
- در console بررسی کنید که خطایی وجود ندارد

### مشکل: متریک‌ها به‌روز نمی‌شوند

- بررسی کنید که `interval` مقدار مناسبی دارد
- در Chrome DevTools بررسی کنید که `performance.memory` موجود است

### مشکل: مصرف CPU دقیق نیست

- ردیابی CPU در مرورگر تخمینی است
- برای دقت بیشتر از Chrome DevTools Performance tab استفاده کنید

---

## 📚 منابع بیشتر

- [React Profiler API](https://react.dev/reference/react/Profiler)
- [Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance)
- [Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/)

---

## 💡 مثال کامل

```jsx
// src/pages/dashboard/orthodontics/patient/[id].jsx
import { Helmet } from 'react-helmet-async';
import { CONFIG } from 'src/config-global';
import { PatientOrthodonticsView } from 'src/sections/orthodontics/patient/view';
import { PerformanceMonitor } from 'src/components/performance-monitor';

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{`مدیریت بیمار - ${CONFIG.appName}`}</title>
      </Helmet>

      <PatientOrthodonticsView />
      
      {/* Performance Monitor */}
      {import.meta.env.DEV && (
        <PerformanceMonitor 
          componentName="PatientOrthodonticsView" 
          position="bottom-right"
          showOnMount={false}
        />
      )}
    </>
  );
}
```

---

**نکته**: برای استفاده در production، حتماً مانیتورها را غیرفعال کنید یا فقط در حالت development نمایش دهید.


