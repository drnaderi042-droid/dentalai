# اصلاح Critical Bug: اعمال Sigmoid

## 🔍 مشکل

نتایج بعد از soft-argmax بدتر شد (MRE: 4.79mm → 28.84mm)!

## ✅ علت

در کد evaluation و inference از **sigmoid** استفاده می‌شود:
- `evaluate.py` خط 63: `heatmaps = torch.sigmoid(outputs_resized).cpu().numpy()`
- `inference.py` خط 110: `heatmaps_np = torch.sigmoid(heatmaps).cpu().numpy()[0]`

اما در `hrnet_production_service.py` **sigmoid اعمال نمی‌شد**!

## 🔧 اصلاح

### تغییرات در `hrnet_production_service.py`:

**قبل:**
```python
heatmaps = self.model(img_tensor)
heatmaps = heatmaps[0]  # (19, H, W)
```

**بعد:**
```python
outputs = self.model(img_tensor)
heatmaps = torch.sigmoid(outputs)  # CRITICAL FIX!
heatmaps = heatmaps[0]  # (19, H, W)
```

### بهبود `heatmap_to_coordinate`:

- استفاده از soft-argmax (همانند `Aariz/utils.py`)
- با temperature scaling = 2.0
- Fallback به argmax اگر heatmap خیلی flat باشد

---

## 📊 نتایج انتظاری

با این اصلاحات باید:
- **MRE**: < 2mm (مشابه checkpoint: 0.63mm)
- **SDR @ 2mm**: > 70%
- دقت sub-pixel با soft-argmax

---

**تاریخ**: 2024-11-01
**وضعیت**: ✅ اصلاح شد - نیاز به تست

