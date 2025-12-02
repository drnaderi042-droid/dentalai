# راهنمای اصلاح کمان‌ها و زاویه‌ها در Canvas

## الگوی کلی ترسیم کمان

### 1. محاسبه زوایای شروع و پایان
```javascript
// محاسبه زاویه از vertex به نقطه اول
let angle1 = Math.atan2(point1.y - vertex.y, point1.x - vertex.x);
// محاسبه زاویه از vertex به نقطه دوم
let angle2 = Math.atan2(point2.y - vertex.y, point2.x - vertex.x);
```

### 2. انتخاب کمان کوچکتر (مهم!)
```javascript
// محاسبه تفاوت زاویه
let angleDiff = angle2 - angle1;
if (angleDiff < 0) angleDiff += 2 * Math.PI;

// اگر تفاوت بیشتر از 180 درجه است، کمان بزرگتر است
// باید جهت را معکوس کنیم تا کمان کوچکتر ترسیم شود
if (angleDiff > Math.PI) {
  const temp = angle1;
  angle1 = angle2;
  angle2 = temp;
}
```

### 3. اطمینان از ترتیب صحیح (endAngle > startAngle)
```javascript
// ctx.arc نیاز دارد که endAngle > startAngle باشد
if (angle2 < angle1) {
  angle2 += 2 * Math.PI;
}
```

### 4. ترسیم کمان
```javascript
ctx.strokeStyle = '#FFD700'; // رنگ کمان
ctx.lineWidth = actualLineWidth;
ctx.setLineDash([]); // خط توپر
ctx.beginPath();
ctx.arc(vertex.x, vertex.y, radius, angle1, angle2);
ctx.stroke();
```

## مشکلات رایج و راه حل

### مشکل 1: کمان در جهت اشتباه ترسیم می‌شود
**راه حل:** بررسی کنید که `angleDiff > Math.PI` را چک کرده‌اید و در صورت نیاز زوایا را معکوس کنید.

### مشکل 2: کمان کامل دایره ترسیم می‌شود
**راه حل:** مطمئن شوید که `angle2 < angle1` را چک کرده‌اید و `angle2 += 2 * Math.PI` اضافه کرده‌اید.

### مشکل 3: کمان بزرگتر به جای کوچکتر ترسیم می‌شود
**راه حل:** همیشه `angleDiff` را محاسبه کنید و اگر `> Math.PI` بود، `angle1` و `angle2` را جابجا کنید.

## مثال کامل (SNA Angle)

```javascript
// 1. محاسبه زوایا
let snaAngle1 = Math.atan2(sPos.y - nPos.y, sPos.x - nPos.x);
let snaAngle2 = Math.atan2(aPos.y - nPos.y, aPos.x - nPos.x);

// 2. انتخاب کمان کوچکتر
let angleDiff = snaAngle2 - snaAngle1;
if (angleDiff < 0) angleDiff += 2 * Math.PI;
if (angleDiff > Math.PI) {
  const temp = snaAngle1;
  snaAngle1 = snaAngle2;
  snaAngle2 = temp;
}

// 3. اطمینان از ترتیب صحیح
if (snaAngle2 < snaAngle1) {
  snaAngle2 += 2 * Math.PI;
}

// 4. ترسیم
ctx.beginPath();
ctx.arc(nPos.x, nPos.y, radius, snaAngle1, snaAngle2);
ctx.stroke();
```

## نکات مهم

1. **همیشه کمان کوچکتر را ترسیم کنید** - برای زوایای کمتر از 180 درجه
2. **Math.atan2** زاویه را به رادیان برمی‌گرداند (0 تا 2π)
3. **ctx.arc** نیاز دارد که `endAngle > startAngle` باشد
4. **برای زوایای منفی** از `angle += 2 * Math.PI` استفاده کنید

## محل فایل
`vite-js/src/components/advanced-cephalometric-visualizer/advanced-cephalometric-visualizer.jsx`

## جستجوی کمان‌های مشکل‌دار
برای پیدا کردن کمان‌های مشکل‌دار، دنبال `ctx.arc` بگردید و الگوی بالا را اعمال کنید.

