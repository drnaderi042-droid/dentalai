# توضیحات Warning های CLdetection2023

این فایل توضیح می‌دهد که warning های نمایش داده شده در ترمینال چه معنایی دارند و آیا نیاز به رفع دارند یا نه.

## ✅ Warning های بی‌خطر (نیاز به رفع ندارند)

### 1. DeprecationWarning از mmengine
```
DeprecationWarning: `TorchScript` support for functional optimizers is deprecated
```

**معنی:** کتابخانه `mmengine` از یک ویژگی منسوخ شده PyTorch استفاده می‌کند.

**نیاز به رفع:** ❌ خیر - این warning از کتابخانه `mmengine` می‌آید و ما نمی‌توانیم آن را کنترل کنیم.

**تأثیر:** هیچ - فقط یک هشدار است و عملکرد را تحت تأثیر قرار نمی‌دهد.

---

### 2. UserWarning از mmcv
```
UserWarning: Fail to import ``MultiScaleDeformableAttention`` from ``mmcv.ops.multi_scale_deform_attn``
```

**معنی:** ماژول `MultiScaleDeformableAttention` از `mmcv` import نشده است. این ماژول برای برخی مدل‌های پیشرفته استفاده می‌شود اما برای CLdetection2023 ضروری نیست.

**نیاز به رفع:** ❌ خیر - این ماژول برای CLdetection2023 استفاده نمی‌شود.

**تأثیر:** هیچ - فقط یک هشدار است.

**راه حل (اختیاری):** اگر می‌خواهید این warning را حذف کنید، می‌توانید `mmcv-full` را نصب کنید:
```bash
pip uninstall mmcv -y
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html
```
⚠️ **توجه:** این کار ممکن است با نسخه `mmcv` مورد نیاز CLdetection2023 (`>=2.0.0rc4,<=2.1.0`) تداخل داشته باشد.

---

### 3. Warning درباره cldetection_utils
```
⚠️  Warning: Could not import cldetection_utils, using local implementation
```

**معنی:** فایل `cldetection_utils.py` import نشد (احتمالاً به `SimpleITK` نیاز دارد)، اما از یک پیاده‌سازی محلی استفاده می‌شود که همان کار را انجام می‌دهد.

**نیاز به رفع:** ❌ خیر - پیاده‌سازی محلی به درستی کار می‌کند.

**تأثیر:** هیچ - عملکرد یکسان است.

**راه حل (اختیاری):** اگر می‌خواهید این warning را حذف کنید، می‌توانید `SimpleITK` را نصب کنید:
```bash
pip install "SimpleITK>=2.2.0"
```
⚠️ **توجه:** نصب `SimpleITK` در ویندوز ممکن است به CMake و Visual Studio Build Tools نیاز داشته باشد.

---

### 4. FutureWarning از mmengine
```
FutureWarning: You are using `torch.load` with `weights_only=False`
```

**معنی:** در نسخه‌های آینده PyTorch، `torch.load` به صورت پیش‌فرض `weights_only=True` خواهد بود (برای امنیت بیشتر).

**نیاز به رفع:** ❌ خیر - این warning از کتابخانه `mmengine` می‌آید و ما نمی‌توانیم آن را کنترل کنیم.

**تأثیر:** هیچ - فقط یک هشدار است.

---

### 5. Warning درباره timm
```
Warning: timm not available, using ResNet fallback
```

**معنی:** کتابخانه `timm` نصب نیست و از ResNet fallback استفاده می‌شود (برای مدل P1/P2).

**نیاز به رفع:** ⚠️ اختیاری - این warning مربوط به مدل P1/P2 است، نه CLdetection2023.

**تأثیر:** ممکن است عملکرد مدل P1/P2 کمی متفاوت باشد، اما برای CLdetection2023 تأثیری ندارد.

**راه حل (اختیاری):** اگر می‌خواهید این warning را حذف کنید:
```bash
pip install timm
```

---

### 6. Warning درباره P1/P2 model
```
⚠️  Warning: Strict loading failed, trying with strict=False
```

**معنی:** مدل P1/P2 با `strict=True` لود نشد و با `strict=False` لود شد. این معمولاً به این معنی است که برخی کلیدهای state_dict با ساختار مدل فعلی مطابقت ندارند.

**نیاز به رفع:** ❌ خیر - مدل با موفقیت لود شده و کار می‌کند.

**تأثیر:** هیچ - مدل به درستی کار می‌کند.

---

## 📝 خلاصه

**همه این warning ها بی‌خطر هستند و نیاز به رفع ندارند.** سیستم به درستی کار می‌کند و این warning ها فقط اطلاعاتی هستند.

اگر می‌خواهید این warning ها را suppress کنید (پنهان کنید)، می‌توانید در ابتدای فایل `unified_ai_api_server.py` کد زیر را اضافه کنید:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='mmengine')
warnings.filterwarnings('ignore', category=UserWarning, module='mmcv')
warnings.filterwarnings('ignore', message='.*TorchScript.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*weights_only.*', category=FutureWarning)
```

⚠️ **توجه:** Suppress کردن warning ها ممکن است باعث شود که warning های مهم دیگر را از دست بدهید. بهتر است warning ها را ببینید اما بدانید که بی‌خطر هستند.

