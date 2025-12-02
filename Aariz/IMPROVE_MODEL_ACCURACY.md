# 🎯 راهنمای بهبود دقت مدل Aariz

## 📊 وضعیت فعلی مدل

از نتایج evaluation:
- **MRE**: 1.67mm (خوب اما قابل بهبود)
- **SDR @ 2mm**: 71.86% (هدف: > 75%)
- **SDR @ 4mm**: 92.85% (خوب)
- برخی لندمارک‌ها خطای بالایی دارند (تا 3mm)

## 🔍 تحلیل مشکل

اگر لندمارک‌ها در موقعیت نادرست هستند، دلایل احتمالی:

1. **مدل به اندازه کافی آموزش ندیده** - نیاز به epochs بیشتر
2. **Learning rate نامناسب** - ممکن است خیلی بالا یا پایین باشد
3. **Data augmentation ناکافی یا بیش از حد**
4. **Loss function نیاز به تنظیم دارد**
5. **Checkpoint استفاده شده بهترین نیست**

---

## 🚀 راه‌حل‌های بهبود

### راه‌حل 1: استفاده از Checkpoint بهتر

#### بررسی تمام checkpoint ها:

```python
# بررسی MRE هر checkpoint
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_50.pth
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_100.pth
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_150.pth
# ... و غیره
```

#### استفاده از بهترین checkpoint:

```python
# در app_aariz.py یا inference.py
CHECKPOINT_PATH = 'checkpoints/checkpoint_epoch_XXX.pth'  # checkpoint با کمترین MRE
```

---

### راه‌حل 2: Fine-tuning مدل موجود

اگر مدل قبلاً آموزش دیده، می‌توانید آن را fine-tune کنید:

```bash
cd Aariz
python train_optimized.py \
  --model hrnet \
  --resume checkpoints/checkpoint_best.pth \
  --epochs 100 \
  --learning_rate 1e-5 \  # Learning rate پایین‌تر برای fine-tuning
  --mixed_precision \
  --use_ema \
  --gradient_accumulation_steps 2
```

**مزایا:**
- بهبود تدریجی دقت
- حفظ دانش قبلی مدل
- سریع‌تر از آموزش از صفر

---

### راه‌حل 3: آموزش مجدد با تنظیمات بهینه

#### تنظیمات پیشنهادی برای دقت بالاتر:

```bash
python train_optimized.py \
  --model hrnet \
  --dataset_path Aariz \
  --epochs 300 \  # بیشتر از 250
  --batch_size 4 \  # اگر GPU حافظه کافی دارد
  --image_size 512 512 \
  --learning_rate 3e-4 \  # کمی پایین‌تر
  --mixed_precision \
  --use_ema \
  --gradient_accumulation_steps 2 \
  --warmup_epochs 10
```

#### یا با تنظیمات پیشرفته‌تر:

```python
# در config.py یا train_optimized.py
config = {
    'model_name': 'hrnet',
    'image_size': (512, 512),
    'batch_size': 4,
    'epochs': 300,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'heatmap_sigma': 3.5,  # CRITICAL: باید 3-4 باشد
    'focal_alpha': 2.0,
    'focal_beta': 4.0,
    'focal_weight': 0.6,  # افزایش وزن focal loss
    'augmentation': True,
    'rotation_degrees': 3.0,  # کاهش rotation برای دقت بیشتر
    'brightness': 0.05,  # کاهش brightness augmentation
    'contrast': 0.05,
    'scheduler': 'cosine',
    'optimizer': 'adamw',
}
```

---

### راه‌حل 4: Post-Processing برای بهبود نتایج

می‌توانید یک لایه post-processing اضافه کنید:

```python
# در app_aariz.py یا inference.py

def post_process_landmarks(landmarks, image_size):
    """
    Post-processing برای بهبود دقت لندمارک‌ها
    """
    processed = {}
    
    for name, coords in landmarks.items():
        if coords is None:
            processed[name] = None
            continue
        
        x, y = coords['x'], coords['y']
        
        # 1. حذف outliers (اگر خارج از محدوده تصویر باشد)
        if x < 0 or x > image_size[0] or y < 0 or y > image_size[1]:
            # استفاده از median filtering برای لندمارک‌های مجاور
            processed[name] = smooth_with_neighbors(name, landmarks, image_size)
        else:
            processed[name] = coords
    
    # 2. Smoothing برای لندمارک‌های مرتبط (مثلاً دندان‌ها)
    processed = smooth_related_landmarks(processed)
    
    return processed

def smooth_with_neighbors(landmark_name, all_landmarks, image_size):
    """Smooth با استفاده از لندمارک‌های مجاور"""
    # منطق smoothing بر اساس آناتومی
    # ...
    pass
```

---

### راه‌حل 5: بررسی و بهبود داده‌های آموزش

#### بررسی کیفیت annotations:

```python
# اسکریپت بررسی annotations
python -c "
from dataset import CephalometricDataset
import matplotlib.pyplot as plt

dataset = CephalometricDataset('Aariz', 'train', image_size=(512, 512))
# بررسی چند نمونه تصادفی
for i in range(min(10, len(dataset))):
    img, landmarks = dataset[i]
    # بررسی اینکه آیا landmarks منطقی هستند
    print(f'Sample {i}: {len(landmarks)} landmarks')
"
```

#### نکات:
- اطمینان حاصل کنید که annotations درست هستند
- بررسی کنید که pixel size برای هر تصویر درست است
- اگر annotation های متعدد دارید (Senior/Junior)، از بهترین‌ها استفاده کنید

---

### راه‌حل 6: استفاده از Ensemble

می‌توانید چندین مدل را با هم ترکیب کنید:

```python
# ترکیب نتایج چند checkpoint
checkpoints = [
    'checkpoints/checkpoint_best.pth',
    'checkpoints/checkpoint_epoch_200.pth',
    'checkpoints/checkpoint_epoch_250.pth',
]

predictions = []
for ckpt in checkpoints:
    predictor = LandmarkPredictor(ckpt, model_name='hrnet')
    result = predictor.predict(image)
    predictions.append(result['landmarks'])

# متوسط گیری
ensemble_landmarks = {}
for name in predictions[0].keys():
    coords = [p[name] for p in predictions if p[name] is not None]
    if coords:
        ensemble_landmarks[name] = {
            'x': np.mean([c['x'] for c in coords]),
            'y': np.mean([c['y'] for c in coords])
        }
```

---

### راه‌حل 7: تنظیم Heatmap Sigma

`heatmap_sigma` بسیار مهم است! باید 3-4 باشد:

```python
# در config.py
heatmap_sigma: float = 3.5  # CRITICAL: Must be 3-4

# اگر کمتر باشد (مثلاً 1.0):
# - Heatmaps خیلی تیز می‌شوند
# - Model یاد نمی‌گیرد
# - نتایج بدتر می‌شوند
```

---

## 📋 چک‌لیست بهبود

قبل از شروع بهبود:

- [ ] بررسی تمام checkpoint ها و انتخاب بهترین
- [ ] بررسی TensorBoard logs برای دیدن روند آموزش
- [ ] بررسی کیفیت annotations در dataset
- [ ] تست مدل روی چند تصویر sample
- [ ] بررسی اینکه آیا heatmap_sigma = 3.5 است

برای آموزش مجدد:

- [ ] استفاده از `train_optimized.py` (نه `train.py`)
- [ ] فعال کردن `--mixed_precision`
- [ ] فعال کردن `--use_ema`
- [ ] تنظیم `heatmap_sigma = 3.5`
- [ ] استفاده از learning rate مناسب (3e-4 تا 5e-4)
- [ ] آموزش حداقل 250 epochs (ترجیحاً 300)

---

## 🎯 اهداف بهبود

پس از بهبود باید به این اهداف برسید:

- ✅ **MRE < 1.5mm** (در حال حاضر: 1.67mm)
- ✅ **SDR @ 2mm > 75%** (در حال حاضر: 71.86%)
- ✅ **SDR @ 4mm > 95%** (در حال حاضر: 92.85% - خوب است)

---

## 🔧 دستور سریع برای Fine-tuning

```bash
cd Aariz

# Fine-tuning با learning rate پایین
python train_optimized.py \
  --model hrnet \
  --resume checkpoints/checkpoint_best.pth \
  --epochs 100 \
  --learning_rate 1e-5 \
  --mixed_precision \
  --use_ema \
  --batch_size 4 \
  --image_size 512 512

# نتیجه در checkpoints/checkpoint_best.pth ذخیره می‌شود
```

---

## 💡 نکات مهم

1. **صبر کنید**: بهبود دقت زمان‌بر است
2. **Monitor کنید**: از TensorBoard استفاده کنید
3. **Test کنید**: بعد از هر بهبود، روی چند تصویر تست کنید
4. **Save کنید**: بهترین checkpoint را نگه دارید
5. **Document کنید**: تنظیماتی که استفاده می‌کنید را یادداشت کنید

---

## 🆘 اگر مشکل ادامه داشت

1. بررسی logs در TensorBoard
2. بررسی MRE هر لندمارک به صورت جداگانه
3. تست مدل روی تصاویر مختلف
4. بررسی اینکه آیا مشکل از dataset است یا از مدل
5. در نظر گرفتن استفاده از معماری دیگر (مثلاً ResNet یا Hourglass)

---

**موفق باشید! 🎉**

