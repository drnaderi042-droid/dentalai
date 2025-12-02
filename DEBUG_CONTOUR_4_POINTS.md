# Debug: چرا فقط 4 نقطه به جای 60-250 نقطه؟

## مشکل
بعد از اصلاح landmark mapping، هنوز تعداد نقاط خیلی کم است:
- اکثر نواحی: **4 نقطه** ❌
- فقط `orbital_rim` و `sella_turcica`: **48 نقطه** ✅

## تغییرات انجام‌شده برای Debug

### 1. حذف نواحی upper_teeth و lower_teeth
این نواحی حذف شدند چون:
- فقط 1 landmark دارند (U1 یا L1)
- نمی‌توانند contour معنی‌داری تولید کنند
- کاربرد محدود در تشخیص

### 2. اضافه کردن Debug Logging
Logging اضافه شد به `contour_detection_service.py`:

```python
# خط 791: نمایش تعداد seed points
print(f"[DEBUG] {region_type}: Found {len(seed_points)} seed points from landmarks: {config['landmarks']}")

# خط 839: نمایش تعداد نقاط بعد از RyanChoi
print(f"[OK] Used RyanChoi advanced contour detection for {region_type} - Got {len(contour_original) if contour_original is not None else 0} points")

# خط 875: نمایش تعداد نقاط قبل از simplification
print(f"[DEBUG] {region_type}: contour has {len(contour_original)} points before simplification")
```

## چگونه تست کنیم؟

### گام 1: Restart سرور Aariz
```bash
# Ctrl+C برای متوقف کردن
cd cephx_service
python app_aariz.py
```

### گام 2: تست در UI
1. به `/dashboard/ai-model-test` بروید
2. تصویر آپلود کنید
3. سوئیچ Contours را فعال کنید
4. دکمه "شروع تست" بزنید

### گام 3: بررسی Terminal Logs
در terminal Aariz باید ببینید:

```
[DEBUG] soft_tissue_profile: Found 6 seed points from landmarks: ['N', 'Sn', 'UL', 'LL', 'PogSoft', 'Me']
[OK] Used RyanChoi advanced contour detection for soft_tissue_profile - Got 250 points
[DEBUG] soft_tissue_profile: contour has 250 points before simplification
```

## سناریوهای مختلف و تشخیص مشکل

### سناریو 1: seed_points کم است (2-3 نقطه)
```
[DEBUG] mandible_border: Found 2 seed points from landmarks: ['Go', 'Gn', 'Me', 'Pog']
```
**علت**: برخی landmarks وجود ندارند در خروجی HRNet
**راه‌حل**: بررسی کنیم که آیا این landmarks واقعاً در دیکشنری landmarks هستند

### سناریو 2: RyanChoi fail می‌کند
```
[ERROR] RyanChoi contour detection failed: ...
[OK] Used Enhanced (fallback) for mandible_border
```
**علت**: خطا در RyanChoi service
**راه‌حل**: بررسی exception و رفع مشکل

### سناریو 3: RyanChoi فقط seed_points برمی‌گرداند
```
[OK] Used RyanChoi advanced contour detection for soft_tissue_profile - Got 4 points
```
**علت**: RyanChoi به جای تولید contour کامل، فقط seed_points را برگردانده
**راه‌حل**: بررسی کد RyanChoi - احتمالاً active contour fail می‌کند

### سناریو 4: simplification خیلی زیاد است
```
[DEBUG] soft_tissue_profile: contour has 500 points before simplification
[DEBUG] soft_tissue_profile: contour has 4 points after simplification
```
**علت**: پارامترهای simplification خیلی aggressive هستند
**راه‌حل**: کاهش `simplify_epsilon` در REGION_CONFIGS

## احتمالات برای 4 نقطه

### چرا دقیقاً 4 نقطه؟
4 نقطه احتمالاً به این معنی است:
1. **Simplification زیاد**: `simplify_contour()` با `epsilon` بالا باعث شده که کانتور به 4 نقطه ساده شود
2. **Fallback به seed_points**: اگر active contour fail کند، فقط seed_points (حدود 2-6 نقطه) برمی‌گردند، که بعد از simplification به 4 نقطه می‌رسند

### پارامترهای مشکوک
```python
'simplify_epsilon': 1.2,  # soft_tissue_profile
'simplify_epsilon': 1.5,  # lips
'simplify_epsilon': 2.0,  # mandible, maxilla
'simplify_epsilon': 2.5,  # orbital_rim
```

برای تست، می‌توانیم این مقادیر را کاهش دهیم:
- از 1.2 به 0.5
- از 2.0 به 0.8
- از 2.5 به 1.0

## نکات مهم

### چرا orbital_rim و sella_turcica 48 نقطه دارند؟
چون از **circle/ellipse fitting** استفاده می‌کنند:
```python
'use_circle_fit': True,
'circle_fit_points': 48,
```

این روش مستقیماً 48 نقطه تولید می‌کند بدون simplification!

### پیشنهاد راه‌حل
1. **کاهش simplify_epsilon**: از 2.0 به 0.5
2. **افزایش max_points**: از 150 به 300
3. **غیرفعال کردن simplification**: برای تست موقت

## آزمایش راه‌حل

برای تست سریع، می‌توانیم `simplify_epsilon` را در `REGION_CONFIGS` تغییر دهیم:

```python
# قبل
'simplify_epsilon': 2.0,

# بعد (برای تست)
'simplify_epsilon': 0.5,  # کاهش simplification
```

یا کاملاً غیرفعال کنیم:
```python
# در تابع detect_contour، خط ~895
# کامنت کنیم:
# epsilon = config.get('simplify_epsilon', 2.0)
# contour_original = self.simplify_contour(contour_original, epsilon)
```

---

**نکته**: ابتدا با logging ببینیم که واقعاً چه اتفاقی می‌افتد، بعد تصمیم بگیریم!



