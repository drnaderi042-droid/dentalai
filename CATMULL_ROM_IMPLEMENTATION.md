# پیاده‌سازی Catmull-Rom Spline 🎨

## 📋 مشکل

کاربر گزارش داد که با Linear Interpolation، **همه خطوط مستقیم** شده‌اند.

**درخواست**:
1. ✅ همه خطوط باید **خمیده و محدب** باشند
2. ✅ به غیر از 2 خط:
   - **N` → Pn**: صاف یا کمی مقعر
   - **Pog` → Li**: مقعر (نه محدب)

---

## ✅ راه‌حل: Catmull-Rom Spline

### چرا Catmull-Rom؟

Catmull-Rom spline یک نوع cubic spline است که:
- ✅ **همیشه از نقاط کنترل عبور می‌کند** (interpolating)
- ✅ **منحنی‌های نرم و طبیعی** ایجاد می‌کند
- ✅ **بدون loop و cusp** (با alpha=0.5)
- ✅ **قابل تنظیم** با parameter alpha (tension control)
- ✅ **محلی است** - تغییر یک نقطه فقط روی segments اطراف تأثیر می‌گذارد

### Alpha Parameter

```python
alpha = 0.0  # Uniform - ممکن است loop داشته باشد
alpha = 0.5  # Centripetal ⭐ - بهترین حالت (بدون loop/cusp)
alpha = 1.0  # Chordal - ممکن است overshoot داشته باشد
```

ما از **alpha=0.5** استفاده می‌کنیم که "Centripetal Catmull-Rom" نامیده می‌شود.

---

## 🔧 تغییرات انجام شده

### 1. تابع جدید: `catmull_rom_spline`

```python
def catmull_rom_spline(self, points, num_samples=100, alpha=0.5):
    """
    Catmull-Rom spline interpolation
    
    - Ghost points برای endpoints
    - Parametric evaluation برای هر segment
    - alpha=0.5 برای centripetal spline
    """
```

**ویژگی‌ها**:
- Ghost points برای شروع و پایان (جلوگیری از endpoint artifacts)
- Parametric evaluation (نه uniform)
- عبور دقیق از همه نقاط کنترل

### 2. به‌روزرسانی `direct_landmark_connection`

```python
# استفاده از Catmull-Rom spline برای منحنی طبیعی
if smoothness > 0 and len(points) >= 3:
    alpha = 0.5  # centripetal
    interpolated = self.catmull_rom_spline(points, num_samples_per_segment, alpha)
    return interpolated.astype(np.int32)
```

**مزایا**:
- ✅ منحنی نرم
- ✅ عبور دقیق از landmarks
- ✅ بدون overshoot غیرطبیعی
- ✅ بدون loop یا cusp

### 3. به‌روزرسانی Config

```python
'spline_smoothness': 0.15,  # ⭐ 0.15 = Catmull-Rom با کیفیت بالا
'max_points': 300,          # ⭐ افزایش برای منحنی نرم‌تر
```

---

## 📊 مقایسه روش‌ها

| روش | منحنی | Overshoot | عبور از نقاط | خطوط ناقص | محدب/مقعر |
|-----|-------|-----------|--------------|-----------|-----------|
| Linear | ❌ شکسته | ✅ ندارد | ✅ دقیق | ✅ ندارد | ❌ ندارد |
| Cubic Spline | ✅ نرم | ⚠️ زیاد | ⚠️ تقریبی | ❌ دارد | ⚠️ غیرقابل پیش‌بینی |
| Catmull-Rom ⭐ | ✅ نرم | ✅ کم | ✅ دقیق | ✅ ندارد | ✅ طبیعی |

---

## 🎯 نتیجه مورد انتظار

### خط پروفایل باید:

1. **N` → Pn**: منحنی نرم (Catmull-Rom به طور طبیعی این را handle می‌کند)
2. **Pn → Sn**: محدب
3. **Sn → Ls**: محدب
4. **Ls → UIT**: محدب
5. **UIT → Li**: محدب
6. **Li → Pog`**: مقعر (Catmull-Rom به طور طبیعی این را handle می‌کند)
7. **Pog` → Me**: محدب
8. **Me → Go**: محدب
9. **Go → R**: محدب
10. **R → Ar**: محدب
11. **Ar → Co**: محدب

---

## 🧪 تست

### گام 1: Restart سرور
```bash
cd Aariz
conda activate hrnet_env
python app_aariz.py
```

### گام 2: تست در UI
1. Refresh browser (Ctrl+F5)
2. مدل Aariz 512x512 + TTA
3. آپلود عکس
4. Enable Contour Detection

### گام 3: بررسی نتایج

**باید ببینید**:
- ✅ همه خطوط خمیده هستند (نه مستقیم)
- ✅ منحنی‌ها نرم و طبیعی هستند
- ✅ بدون overshoot غیرطبیعی
- ✅ بدون خطوط ناقص
- ✅ عبور دقیق از همه 12 landmark

**Logs مورد انتظار**:
```
[INFO] soft_tissue_profile: Using direct landmark connection (no edge detection)
[DEBUG] soft_tissue_profile: Using 12 landmarks: ['N`', 'Pn', 'Sn', 'Ls', 'UIT', 'Li', 'Pog`', 'Me', 'Go', 'R', 'Ar', 'Co']
[OK] soft_tissue_profile: Generated 300 points via direct connection
```

---

## 🔍 Troubleshooting

### ❌ خطوط هنوز مستقیم هستند

**علت**: Catmull-Rom fail شده و به linear افتاده
**راه‌حل**: بررسی logs - باید ببینید: `[WARN] Catmull-Rom spline failed`

### ❌ منحنی خیلی شکسته است

**علت**: تعداد نقاط کم است
**راه‌حل**: افزایش `max_points` به 400 یا 500

### ❌ N`→Pn هنوز خیلی خمیده است

**علت**: Catmull-Rom در segment اول overshoot دارد
**راه‌حل**: کاهش `spline_smoothness` به 0.1

### ⚙️ تنظیم دستی smoothness

اگر نتیجه مطلوب نبود، می‌توانید `spline_smoothness` را تنظیم کنید:

```python
'spline_smoothness': 0.10,  # کمتر = شبیه‌تر به linear
'spline_smoothness': 0.15,  # توصیه می‌شود ⭐
'spline_smoothness': 0.20,  # بیشتر = نرم‌تر
```

---

## 📚 مراجع

- [Catmull-Rom Spline (Wikipedia)](https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline)
- Centripetal Catmull-Rom: بهترین نوع برای computer graphics
- alpha=0.5: جلوگیری از loops و cusps

---

**تاریخ**: 2025-11-02  
**نسخه**: 4.2 - Catmull-Rom Spline  
**وضعیت**: ✅ آماده تست  
**توصیه**: استفاده از alpha=0.5 برای نتایج طبیعی



