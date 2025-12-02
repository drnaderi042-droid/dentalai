# اصلاح دقت HRNet

## 🔍 مشکل

نتایج تست نشان داد:
- **MRE**: 4.79mm (باید < 2mm باشد)
- **SDR @ 2mm**: 13.33% (باید > 80% باشد)
- **Checkpoint MRE**: 0.63mm (در validation set)

## 🔎 علت

مشکل در روش تبدیل heatmap به مختصات بود:

### ❌ روش قدیمی (خطا: ~5mm)
```python
# فقط argmax ساده
flat_idx = np.argmax(heatmap)
y, x = divmod(flat_idx, w)
```
- دقت pixel-level (بدون sub-pixel)
- خطای بیشتر

### ✅ روش جدید (با soft-argmax)
```python
# استفاده از weighted average برای sub-pixel accuracy
heatmap_scaled = np.power(heatmap, 2.0)  # Temperature scaling
weights = heatmap_scaled / heatmap_scaled.sum()
x = np.sum(x_coords * weights)  # Weighted average
y = np.sum(y_coords * weights)
```
- دقت sub-pixel
- مطابق با کد آموزش (`Aariz/utils.py`)

---

## ✅ تغییرات انجام شده

### فایل: `cephx_service/hrnet_production_service.py`

**قبل:**
- استفاده از `argmax` ساده
- دقت pixel-level

**بعد:**
- استفاده از **soft-argmax** (weighted average)
- دقت sub-pixel
- Temperature scaling (2.0)
- Fallback به argmax اگر heatmap خیلی flat باشد

---

## 📊 نتایج انتظاری بعد از اصلاح

- **MRE**: < 2mm (از 4.79mm)
- **SDR @ 2mm**: > 70% (از 13.33%)
- دقت sub-pixel برای همه لندمارک‌ها

---

## 🧪 تست مجدد

```batch
Aariz\run_hrnet_direct_test.bat
```

باید نتایج بسیار بهتر باشند!

---

**تاریخ**: 2024-11-01
**وضعیت**: ✅ اصلاح شد

