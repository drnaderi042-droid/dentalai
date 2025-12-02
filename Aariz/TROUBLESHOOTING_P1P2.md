# 🔧 Troubleshooting P1/P2 Model Integration

## ⚠️ مشکل فعلی:

```
503 SERVICE UNAVAILABLE
Error: Aariz 512 model not available
```

**اما:** این خطا مربوط به **Aariz 512 model** است، نه P1/P2!

---

## ✅ تغییرات انجام شده:

### 1. **Error Handling بهتر:**
- ✅ `strict=False` fallback برای load کردن model
- ✅ Try-except در همه endpointها
- ✅ اگر p1/p2 load نشد، endpointها همچنان کار می‌کنند

### 2. **Debug Logging:**
- ✅ Path checking
- ✅ Detailed error messages
- ✅ Status reporting

---

## 🔍 بررسی مشکل:

### مشکل 1: Aariz 512 Model در دسترس نیست

**خطا:**
```
503 - Aariz 512 model not available
```

**راه‌حل:**
```bash
# بررسی کنید که Aariz model موجود است:
ls Aariz/checkpoint_best_512.pth

# اگر موجود نیست، باید train کنید یا از جای دیگری کپی کنید
```

### مشکل 2: P1/P2 Model Keys Mismatch

**خطا:**
```
Missing keys: ...
Unexpected keys: ...
```

**راه‌حل:**
✅ **قبلاً fix شده!** با `strict=False` fallback

### مشکل 3: Import Error

**خطا:**
```
ImportError: cannot import name 'HRNetP1P2HeatmapDetector'
```

**راه‌حل:**
```bash
# بررسی کنید که فایل موجود است:
ls aariz/model_heatmap.py

# بررسی کنید که path درست است:
python -c "import sys; print(sys.path)"
```

---

## 🧪 تست Model Loading:

```bash
cd aariz
python test_model_load.py
```

**انتظار:**
```
[OK] All tests passed! Model can be loaded successfully.
```

اگر این کار کرد، model درست است و مشکل از unified_ai_api_server است.

---

## 🔧 Debugging Unified API Server:

### 1. بررسی Path:

```python
# در unified_ai_api_server.py
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'aariz', 'models', 'hrnet_p1p2_heatmap_best.pth')
print(f"Model path: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")
```

### 2. بررسی Import:

```python
# در unified_ai_api_server.py
aariz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aariz')
print(f"Aariz dir: {aariz_dir}")
print(f"Exists: {os.path.exists(aariz_dir)}")
print(f"Model file exists: {os.path.exists(os.path.join(aariz_dir, 'model_heatmap.py'))}")
```

### 3. بررسی Model File:

```bash
# در root directory
ls aariz/models/hrnet_p1p2_heatmap_best.pth
```

---

## 📋 چک‌لیست:

- [ ] `aariz/model_heatmap.py` موجود است
- [ ] `aariz/models/hrnet_p1p2_heatmap_best.pth` موجود است
- [ ] `timm` نصب است (`pip install timm`)
- [ ] `torch` نصب است
- [ ] `Aariz/checkpoint_best_512.pth` موجود است (برای endpointهای detect)

---

## 🚀 راه‌حل سریع:

### اگر P1/P2 Model مشکل دارد:

1. **غیرفعال کردن موقت:**
   - Comment کردن `detect_p1p2_for_landmarks` calls
   - Endpointها بدون p1/p2 کار می‌کنند

2. **بررسی Logs:**
   ```bash
   # در terminal که unified_ai_api_server را اجرا می‌کنید
   # خطاهای مربوط به p1/p2 را ببینید
   ```

3. **تست مستقیم:**
   ```bash
   cd aariz
   python test_model_load.py
   ```

### اگر Aariz Model مشکل دارد:

```bash
# بررسی کنید که model موجود است:
ls Aariz/checkpoint_best_512.pth
ls Aariz/checkpoint_best_768.pth

# اگر موجود نیست، باید train کنید
```

---

## 📊 Status Codes:

| Status | معنی |
|--------|------|
| `not_loaded` | هنوز load نشده (lazy loading) |
| `ready` | آماده استفاده |
| `model_not_found` | فایل model موجود نیست |
| `import_error` | مشکل در import |
| `error: ...` | خطای دیگر |

---

## 🔄 Workflow Debugging:

```
1. Request → /detect-512
2. Load Aariz 512 model
   ├─ Success → Continue
   └─ Fail → Return 503 ❌ (مشکل اصلی!)
3. Detect landmarks
4. Try load P1/P2 model
   ├─ Success → Add p1/p2
   └─ Fail → Continue without p1/p2 ✅
5. Return landmarks
```

**مشکل:** مرحله 2 fail می‌شود!

---

## 💡 راه‌حل:

### گام 1: بررسی Aariz Models

```bash
cd "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy"
dir Aariz\checkpoint_best_*.pth
```

### گام 2: اگر موجود نیست

```bash
# باید Aariz models را train کنید یا از جای دیگری کپی کنید
```

### گام 3: Restart Server

```bash
# unified_ai_api_server.py را restart کنید
python unified_ai_api_server.py
```

---

## 📝 خلاصه:

✅ **P1/P2 Model:** می‌تواند load شود (test_model_load.py موفق بود)  
❌ **Aariz 512 Model:** در دسترس نیست (مشکل اصلی)  
✅ **Error Handling:** اضافه شده (p1/p2 optional است)  

**مشکل اصلی:** Aariz 512 model موجود نیست، نه P1/P2!

**راه‌حل:** مطمئن شوید که `Aariz/checkpoint_best_512.pth` موجود است.













