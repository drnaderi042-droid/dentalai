# راهنمای کامل ردیابی عملکرد کامپوننت‌ها

این راهنما نحوه ردیابی مصرف CPU و RAM برای **هر کامپوننت به صورت جداگانه** را توضیح می‌دهد.

## 📋 فهرست مطالب

1. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
2. [استفاده از TrackedComponent](#استفاده-از-trackedcomponent)
3. [استفاده از HOC](#استفاده-از-hoc)
4. [مثال‌های عملی](#مثال‌های-عملی)
5. [نمایش نتایج](#نمایش-نتایج)

---

## نصب و راه‌اندازی

تمام فایل‌های لازم ایجاد شده است و `PerformanceProvider` در `app.jsx` اضافه شده است.

---

## استفاده از TrackedComponent

### روش 1: Wrap کردن مستقیم

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';

function MyComponent() {
  return (
    <TrackedComponent componentName="MyComponent">
      <div>محتوای کامپوننت</div>
    </TrackedComponent>
  );
}
```

### روش 2: Wrap کردن چند کامپوننت

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';

function MyPage() {
  return (
    <>
      <TrackedComponent componentName="Header">
        <Header />
      </TrackedComponent>
      
      <TrackedComponent componentName="Sidebar">
        <Sidebar />
      </TrackedComponent>
      
      <TrackedComponent componentName="MainContent">
        <MainContent />
      </TrackedComponent>
    </>
  );
}
```

---

## استفاده از HOC

### روش 1: استفاده مستقیم

```jsx
import { withTrackedComponent } from 'src/components/performance-monitor';

function MyComponent() {
  return <div>محتوای کامپوننت</div>;
}

// Wrap کردن کامپوننت
export default withTrackedComponent(MyComponent, {
  componentName: 'MyComponent',
  interval: 1000,
  trackMemory: true,
  trackCPU: true,
});
```

### روش 2: استفاده با export

```jsx
import { withTrackedComponent } from 'src/components/performance-monitor';

function MyComponent() {
  return <div>محتوای کامپوننت</div>;
}

const TrackedMyComponent = withTrackedComponent(MyComponent, {
  componentName: 'MyComponent',
});

export { TrackedMyComponent as MyComponent };
```

---

## مثال‌های عملی

### مثال 1: ردیابی کامپوننت‌های صفحه Patient

```jsx
// src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx
import { TrackedComponent } from 'src/components/performance-monitor';

export function PatientOrthodonticsView() {
  return (
    <>
      <TrackedComponent componentName="PatientHeader">
        <PatientHeader />
      </TrackedComponent>
      
      <TrackedComponent componentName="PatientTabs">
        <CustomTabs tabs={navigationTabs} />
      </TrackedComponent>
      
      <TrackedComponent componentName="PatientImages">
        <PatientImages images={uploadedImages} />
      </TrackedComponent>
      
      <TrackedComponent componentName="AIDiagnosisDisplay">
        <AIDiagnosisDisplay />
      </TrackedComponent>
    </>
  );
}
```

### مثال 2: ردیابی کامپوننت‌های داخلی

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';

function ImageGallery({ images }) {
  return (
    <TrackedComponent componentName="ImageGallery">
      <Grid container>
        {images.map((image) => (
          <TrackedComponent key={image.id} componentName={`ImageCard-${image.id}`}>
            <ImageCard image={image} />
          </TrackedComponent>
        ))}
      </Grid>
    </TrackedComponent>
  );
}
```

### مثال 3: ردیابی دکمه‌ها و فرم‌ها

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';

function MyForm() {
  return (
    <form>
      <TrackedComponent componentName="FormFields">
        <TextField label="نام" />
        <TextField label="ایمیل" />
      </TrackedComponent>
      
      <TrackedComponent componentName="SubmitButton">
        <Button type="submit">ارسال</Button>
      </TrackedComponent>
    </form>
  );
}
```

### مثال 4: ردیابی کامپوننت‌های Lazy Loaded

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';
import { Suspense } from 'react';

const HeavyComponent = React.lazy(() => import('./heavy-component'));

function MyPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <TrackedComponent componentName="HeavyComponent">
        <HeavyComponent />
      </TrackedComponent>
    </Suspense>
  );
}
```

---

## نمایش نتایج

### استفاده از Advanced Performance Monitor

```jsx
import { AdvancedPerformanceMonitor } from 'src/components/performance-monitor';

export default function Page() {
  return (
    <>
      <YourComponents />
      
      {import.meta.env.DEV && (
        <AdvancedPerformanceMonitor
          showTreeView={true}
          showDetailsPanel={true}
          treeViewPosition="bottom-left"
          detailsPanelPosition="bottom-right"
        />
      )}
    </>
  );
}
```

### استفاده جداگانه

```jsx
import { 
  PerformanceTreeView, 
  PerformanceDetailsPanel 
} from 'src/components/performance-monitor';

export default function Page() {
  const [selectedComponent, setSelectedComponent] = useState(null);
  
  return (
    <>
      <YourComponents />
      
      {import.meta.env.DEV && (
        <>
          <PerformanceTreeView
            position="bottom-left"
            onComponentSelect={setSelectedComponent}
          />
          <PerformanceDetailsPanel
            componentName={selectedComponent}
            position="bottom-right"
          />
        </>
      )}
    </>
  );
}
```

---

## تنظیمات TrackedComponent

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `componentName` | string | required | نام کامپوننت برای شناسایی |
| `interval` | number | 1000 | فاصله به‌روزرسانی (ms) |
| `trackMemory` | boolean | true | ردیابی حافظه |
| `trackCPU` | boolean | true | ردیابی CPU |
| `logProfiler` | boolean | false | لاگ کردن Profiler |

### مثال با تنظیمات

```jsx
<TrackedComponent
  componentName="MyComponent"
  options={{
    interval: 2000,      // هر 2 ثانیه به‌روزرسانی
    trackMemory: true,    // ردیابی حافظه
    trackCPU: true,       // ردیابی CPU
    logProfiler: true,    // لاگ Profiler در console
  }}
>
  <MyComponent />
</TrackedComponent>
```

---

## نکات مهم

1. **نام‌گذاری**: از نام‌های واضح و منحصر به فرد استفاده کنید
2. **سطح ردیابی**: فقط کامپوننت‌های مهم را ردیابی کنید
3. **Performance Impact**: ردیابی خودش کمی overhead دارد
4. **Development Only**: فقط در حالت development استفاده کنید

---

## مثال کامل: صفحه Patient

```jsx
// src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx
import { TrackedComponent } from 'src/components/performance-monitor';

export function PatientOrthodonticsView() {
  return (
    <Container>
      {/* Header */}
      <TrackedComponent componentName="PatientHeader">
        <Box sx={{ mb: 3 }}>
          <Typography variant="h4">اطلاعات بیمار</Typography>
        </Box>
      </TrackedComponent>
      
      {/* Tabs */}
      <TrackedComponent componentName="PatientTabs">
        <CustomTabs
          tabs={navigationTabs}
          currentTab={currentTab}
          onChange={setCurrentTab}
        />
      </TrackedComponent>
      
      {/* Images Grid */}
      <TrackedComponent componentName="ImagesGrid">
        <Grid container spacing={2}>
          {uploadedImages.map((image) => (
            <Grid item key={image.id} xs={12} sm={6} md={4}>
              <TrackedComponent componentName={`ImageCard-${image.id}`}>
                <ImageCard image={image} />
              </TrackedComponent>
            </Grid>
          ))}
        </Grid>
      </TrackedComponent>
      
      {/* AI Diagnosis */}
      {currentTab === 'diagnosis' && (
        <TrackedComponent componentName="AIDiagnosis">
          <AIDiagnosisDisplay patientId={id} />
        </TrackedComponent>
      )}
    </Container>
  );
}
```

---

## عیب‌یابی

### مشکل: کامپوننت در Tree View نمایش داده نمی‌شود

- مطمئن شوید که `PerformanceProvider` در `app.jsx` اضافه شده است
- بررسی کنید که کامپوننت با `TrackedComponent` wrap شده است
- نام کامپوننت باید منحصر به فرد باشد

### مشکل: متریک‌ها به‌روز نمی‌شوند

- بررسی کنید که `interval` مقدار مناسبی دارد
- مطمئن شوید که کامپوننت mount شده است

### مشکل: Details Panel خالی است

- یک کامپوننت را از Tree View انتخاب کنید
- مطمئن شوید که کامپوننت ردیابی شده است

---

## خلاصه

1. کامپوننت‌های مورد نظر را با `TrackedComponent` wrap کنید
2. `AdvancedPerformanceMonitor` را در صفحه اضافه کنید
3. در Tree View کامپوننت‌ها را ببینید
4. روی هر کامپوننت کلیک کنید تا جزئیات را ببینید

**نکته**: برای بهترین نتیجه، کامپوننت‌های مهم و سنگین را ردیابی کنید.


