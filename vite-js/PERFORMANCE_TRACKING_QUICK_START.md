# راهنمای سریع: ردیابی عملکرد هر کامپوننت

## 🚀 شروع سریع

### مرحله 1: Wrap کردن کامپوننت‌ها

کامپوننت‌های مورد نظر را با `TrackedComponent` wrap کنید:

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';

// مثال: Wrap کردن یک کامپوننت
<TrackedComponent componentName="MyButton">
  <Button>کلیک کنید</Button>
</TrackedComponent>

// مثال: Wrap کردن یک Card
<TrackedComponent componentName="PatientCard">
  <Card>
    <CardContent>...</CardContent>
  </Card>
</TrackedComponent>

// مثال: Wrap کردن یک Image
<TrackedComponent componentName="PatientImage">
  <img src="..." alt="..." />
</TrackedComponent>
```

### مرحله 2: مشاهده نتایج

صفحه را باز کنید و در گوشه پایین چپ، **Component Tree** را ببینید.

- روی هر کامپوننت کلیک کنید تا جزئیات را ببینید
- در گوشه پایین راست، **Details Panel** جزئیات کامل را نشان می‌دهد

---

## 📝 مثال عملی برای صفحه Patient

### قبل (بدون ردیابی):

```jsx
export function PatientOrthodonticsView() {
  return (
    <Container>
      <Typography variant="h4">اطلاعات بیمار</Typography>
      <CustomTabs tabs={tabs} />
      <Grid container>
        {images.map((img) => (
          <Grid item key={img.id}>
            <ImageCard image={img} />
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}
```

### بعد (با ردیابی):

```jsx
import { TrackedComponent } from 'src/components/performance-monitor';

export function PatientOrthodonticsView() {
  return (
    <Container>
      <TrackedComponent componentName="PatientHeader">
        <Typography variant="h4">اطلاعات بیمار</Typography>
      </TrackedComponent>
      
      <TrackedComponent componentName="PatientTabs">
        <CustomTabs tabs={tabs} />
      </TrackedComponent>
      
      <TrackedComponent componentName="ImagesGrid">
        <Grid container>
          {images.map((img) => (
            <Grid item key={img.id}>
              <TrackedComponent componentName={`ImageCard-${img.id}`}>
                <ImageCard image={img} />
              </TrackedComponent>
            </Grid>
          ))}
        </Grid>
      </TrackedComponent>
    </Container>
  );
}
```

---

## 🎯 کامپوننت‌های پیشنهادی برای ردیابی

### کامپوننت‌های سنگین:
- ✅ تصاویر بزرگ
- ✅ جداول با داده‌های زیاد
- ✅ چارت‌ها و نمودارها
- ✅ فرم‌های پیچیده
- ✅ کامپوننت‌های Lazy Loaded

### کامپوننت‌های ساده (نیازی به ردیابی نیست):
- ❌ دکمه‌های ساده
- ❌ متن‌های ساده
- ❌ آیکون‌ها

---

## 💡 نکات

1. **نام‌گذاری**: از نام‌های واضح استفاده کنید
   - ✅ `PatientHeader`
   - ✅ `ImageCard-123`
   - ❌ `Component1`
   - ❌ `Card`

2. **سطح ردیابی**: فقط کامپوننت‌های مهم را ردیابی کنید
   - ✅ کامپوننت‌های اصلی صفحه
   - ✅ کامپوننت‌های سنگین
   - ❌ هر کامپوننت کوچک

3. **Performance**: ردیابی خودش کمی overhead دارد
   - فقط در development استفاده کنید
   - فقط کامپوننت‌های مهم را ردیابی کنید

---

## 🔍 مشاهده نتایج

1. صفحه را باز کنید: `http://localhost:3030/dashboard/orthodontics/patient/[id]`
2. در گوشه پایین چپ، **Component Tree** را ببینید
3. روی هر کامپوننت کلیک کنید
4. در گوشه پایین راست، **Details Panel** جزئیات را نشان می‌دهد:
   - مصرف RAM (MB و درصد)
   - مصرف CPU (درصد)
   - زمان رندر (ms)
   - اطلاعات Profiler

---

## 📚 مستندات کامل

برای اطلاعات بیشتر، فایل `PERFORMANCE_TRACKING_GUIDE.md` را مطالعه کنید.


