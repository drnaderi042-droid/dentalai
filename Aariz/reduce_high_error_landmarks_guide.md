# راهنمای کاهش خطای لندمارک‌های با بیشترین خطا

## 📊 لندمارک‌های مشکل‌دار (از تحلیل قبلی)

| رتبه | لندمارک | میانگین خطا (mm) | توضیحات |
|------|---------|------------------|----------|
| 1 | **UMT** (Upper Molar Tip) | 3.805 | نوک دندان آسیای بزرگ بالا |
| 2 | **UPM** (Upper Premolar) | 3.486 | دندان آسیای کوچک بالا |
| 3 | **R** (Ramus point) | 3.331 | نقطه شاخه فک |
| 4 | **Ar** (Articulare) | 2.645 | نقطه مفصل فک |
| 5 | **Go** (Gonion) | 2.618 | زاویه فک پایین |
| 6 | **LMT** (Lower Molar Tip) | 2.545 | نوک دندان آسیای بزرگ پایین |

## 🎯 راهکارهای کاهش خطا

### 1. استفاده از Weighted Loss (پیشنهادی - سریع‌ترین)

افزایش وزن loss برای لندمارک‌های مشکل‌دار در تابع loss.

**مزایا:**
- پیاده‌سازی ساده
- نیاز به تغییرات کم در کد
- تأثیر سریع

**نحوه پیاده‌سازی:**

```python
# در train2.py یا train.py

# تعریف وزن‌ها برای هر لندمارک
LANDMARK_WEIGHTS = {
    'UMT': 2.0,  # وزن بالا برای مشکل‌دارترین
    'UPM': 2.0,
    'R': 1.8,
    'Ar': 1.6,
    'Go': 1.6,
    'LMT': 1.5,
    # بقیه لندمارک‌ها وزن 1.0 دارند (پیش‌فرض)
}

def calculate_weighted_loss(outputs, targets, landmark_symbols, criterion):
    """
    محاسبه weighted loss برای لندمارک‌های مشکل‌دار
    """
    batch_size = outputs.shape[0]
    num_landmarks = outputs.shape[1]
    
    total_loss = 0.0
    
    for i in range(num_landmarks):
        landmark_name = landmark_symbols[i]
        weight = LANDMARK_WEIGHTS.get(landmark_name, 1.0)
        
        # محاسبه loss برای این لندمارک
        landmark_output = outputs[:, i:i+1, :, :]
        landmark_target = targets[:, i:i+1, :, :]
        
        landmark_loss = criterion(landmark_output, landmark_target)
        total_loss += weight * landmark_loss
    
    return total_loss / num_landmarks
```

**استفاده در training:**

```python
# در train_epoch function
if use_adaptive_wing:
    # استفاده از weighted loss
    loss = calculate_weighted_loss(
        outputs_resized, targets, 
        predictor.landmark_symbols, 
        criterion
    )
```

### 2. افزایش Augmentation برای لندمارک‌های مشکل‌دار

افزایش احتمال augmentation برای تصاویری که این لندمارک‌ها را دارند.

**نحوه پیاده‌سازی:**

```python
# در dataset.py

def get_augmented_transforms_for_difficult_landmarks(self):
    """
    Augmentation قوی‌تر برای لندمارک‌های مشکل‌دار
    """
    DIFFICULT_LANDMARKS = ['UMT', 'UPM', 'R', 'Ar', 'Go', 'LMT']
    
    return A.Compose([
        # Rotation بیشتر برای دندانی‌ها
        A.Rotate(limit=15, p=0.7),  # از 10 به 15 درجه
        A.HorizontalFlip(p=0.5),
        
        # Contrast و Brightness بیشتر
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # از 0.2 به 0.3
            contrast_limit=0.3,
            p=0.7
        ),
        
        # Noise بیشتر برای مقاوم‌سازی
        A.GaussNoise(var_limit=(20, 80), p=0.5),  # از (10,50) به (20,80)
        
        # Elastic Transform قوی‌تر
        A.ElasticTransform(
            alpha=150,  # از 120 به 150
            sigma=150*0.05,
            p=0.4
        ),
        
        A.Resize(height=self.image_size[0], width=self.image_size[1]),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
```

### 3. Hard Negative Mining (Focus روی تصاویر مشکل‌دار)

شناسایی تصاویری که در آن‌ها لندمارک‌های مشکل‌دار خطای بالایی دارند و تمرکز بیشتر روی آن‌ها.

**نحوه پیاده‌سازی:**

```python
# اسکریپت برای شناسایی تصاویر مشکل‌دار
def identify_hard_samples(model, val_loader, threshold_mm=3.0):
    """
    شناسایی تصاویری که لندمارک‌های مشکل‌دار خطای بالایی دارند
    """
    hard_samples = []
    DIFFICULT_LANDMARKS = ['UMT', 'UPM', 'R', 'Ar', 'Go', 'LMT']
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # ... پیش‌بینی ...
            
            # محاسبه خطا برای هر لندمارک
            for landmark_idx, landmark_name in enumerate(landmark_symbols):
                if landmark_name in DIFFICULT_LANDMARKS:
                    error_mm = calculate_error(...)
                    if error_mm > threshold_mm:
                        hard_samples.append({
                            'image_id': batch['image_id'],
                            'landmark': landmark_name,
                            'error': error_mm
                        })
    
    return hard_samples

# سپس در training، وزن بیشتر برای این samples:
class WeightedDataset(Dataset):
    def __init__(self, base_dataset, hard_samples, weight_factor=2.0):
        self.base_dataset = base_dataset
        self.hard_samples = {s['image_id']: s for s in hard_samples}
        self.weight_factor = weight_factor
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image_id = sample['image_id']
        
        # اگر sample مشکل‌دار است، تکرار آن
        if image_id in self.hard_samples:
            # می‌توانید sample را duplicate کنید یا augmentation بیشتر اعمال کنید
            pass
        
        return sample
```

### 4. Multi-Scale Training و Testing

Training مدل در چند resolution مختلف و ensemble کردن نتایج.

**مزایا:**
- دقت بالاتر
- مقاوم‌تر در برابر تغییرات scale

**نحوه پیاده‌سازی:**

```python
# در training، استفاده از multi-scale
def train_with_multi_scale(model, train_loader, scales=[512, 768, 1024]):
    for epoch in range(num_epochs):
        for batch in train_loader:
            # انتخاب scale تصادفی
            scale = random.choice(scales)
            
            # Resize image به scale
            image_resized = resize_image(batch['image'], scale)
            
            # Training
            outputs = model(image_resized)
            # ...
```

### 5. Fine-tuning روی Subset مشکل‌دار

Fine-tune کردن مدل فقط روی تصاویری که لندمارک‌های مشکل‌دار دارند.

```python
# ایجاد subset مشکل‌دار
difficult_subset = create_difficult_subset(
    dataset, 
    difficult_landmarks=['UMT', 'UPM', 'R', 'Ar', 'Go', 'LMT'],
    error_threshold=2.5  # mm
)

# Fine-tuning با learning rate پایین
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # LR بسیار پایین
```

### 6. استفاده از Attention Mechanism

اضافه کردن attention layers برای تمرکز بیشتر روی نواحی مشکل‌دار.

```python
# در model.py
class LandmarkAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att
```

### 7. افزایش Resolution برای Training

Training در resolution بالاتر (مثلاً 1024x1024) برای لندمارک‌های کوچک و دقیق.

**مزایا:**
- دقت بالاتر برای لندمارک‌های کوچک
- جزئیات بیشتر

**محدودیت:**
- نیاز به GPU قوی‌تر
- زمان training بیشتر

### 8. استفاده از Ensemble برای لندمارک‌های مشکل‌دار

استفاده از چند مدل و ترکیب نتایج برای لندمارک‌های مشکل‌دار.

```python
def ensemble_predict(models, image, difficult_landmarks):
    """
    Ensemble prediction برای لندمارک‌های مشکل‌دار
    """
    predictions = {}
    
    for model in models:
        pred = model.predict(image)
        for landmark in difficult_landmarks:
            if landmark not in predictions:
                predictions[landmark] = []
            predictions[landmark].append(pred[landmark])
    
    # Average کردن predictions
    final_predictions = {}
    for landmark, preds in predictions.items():
        if preds:
            final_predictions[landmark] = {
                'x': np.mean([p['x'] for p in preds]),
                'y': np.mean([p['y'] for p in preds])
            }
    
    return final_predictions
```

## 📋 اولویت‌بندی راهکارها

### مرحله 1 (سریع‌ترین و مؤثرترین):
1. ✅ **Weighted Loss** - پیاده‌سازی ساده، تأثیر سریع
2. ✅ **افزایش Augmentation** - بهبود generalization

### مرحله 2 (تأثیر متوسط):
3. ✅ **Hard Negative Mining** - تمرکز روی samples مشکل‌دار
4. ✅ **Fine-tuning روی Subset** - بهبود روی موارد خاص

### مرحله 3 (پیچیده‌تر، اما مؤثر):
5. ✅ **Multi-Scale Training** - دقت بالاتر
6. ✅ **افزایش Resolution** - برای لندمارک‌های کوچک

### مرحله 4 (پیشرفته):
7. ✅ **Attention Mechanism** - بهبود architecture
8. ✅ **Ensemble** - ترکیب چند مدل

## 🚀 پیاده‌سازی پیشنهادی

شروع با **Weighted Loss** و **افزایش Augmentation** که سریع‌ترین و مؤثرترین هستند.

### مثال کد کامل برای Weighted Loss:

```python
# در train2.py

# اضافه کردن به ابتدای فایل
DIFFICULT_LANDMARK_WEIGHTS = {
    'UMT': 2.5,   # بیشترین وزن
    'UPM': 2.5,
    'R': 2.0,
    'Ar': 1.8,
    'Go': 1.8,
    'LMT': 1.6,
    'LPM': 1.4,
    'Or': 1.3,
    # بقیه: 1.0 (پیش‌فرض)
}

def calculate_weighted_adaptive_wing_loss(outputs, targets, landmark_symbols, 
                                         base_criterion, device):
    """
    محاسبه weighted adaptive wing loss
    """
    batch_size = outputs.shape[0]
    num_landmarks = outputs.shape[1]
    
    total_loss = 0.0
    
    for i in range(num_landmarks):
        landmark_name = landmark_symbols[i]
        weight = DIFFICULT_LANDMARK_WEIGHTS.get(landmark_name, 1.0)
        
        landmark_output = outputs[:, i:i+1, :, :]
        landmark_target = targets[:, i:i+1, :, :]
        
        # محاسبه loss برای این لندمارک
        landmark_loss = base_criterion(landmark_output, landmark_target)
        
        total_loss += weight * landmark_loss
    
    return total_loss / num_landmarks

# در train_epoch function، جایگزین کردن:
# قبل:
# loss = criterion(outputs_resized, targets)

# بعد:
loss = calculate_weighted_adaptive_wing_loss(
    outputs_resized, targets,
    landmark_symbols,  # باید از dataset بگیرید
    criterion,
    device
)
```

## 📊 انتظارات

بعد از اعمال Weighted Loss و Augmentation:
- **UMT**: از 3.8mm → ~2.5mm (کاهش ~35%)
- **UPM**: از 3.5mm → ~2.3mm (کاهش ~35%)
- **R**: از 3.3mm → ~2.2mm (کاهش ~33%)
- **Overall MRE**: از ~1.6mm → ~1.3mm (کاهش ~20%)

## ⚠️ نکات مهم

1. **متعادل نگه داشتن وزن‌ها**: وزن خیلی بالا ممکن است باعث overfitting شود
2. **Validation monitoring**: همیشه validation loss را monitor کنید
3. **Gradual increase**: به تدریج وزن‌ها را افزایش دهید
4. **A/B Testing**: نتایج را با و بدون weighted loss مقایسه کنید

