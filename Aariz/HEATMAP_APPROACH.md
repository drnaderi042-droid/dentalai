# 🎯 Heatmap-Based Approach - برای دقت زیر 10px

## 🔍 چرا Heatmap بهتر از Direct Regression است؟

### ❌ مشکل Direct Regression:

```
Input Image → Model → [p1_x, p1_y, p2_x, p2_y]
```

**مشکلات:**
1. ❌ Model باید مستقیماً 4 عدد را پیش‌بینی کند
2. ❌ Loss function ساده (MSE) برای landmarks دقیق نیست
3. ❌ Data augmentation مشکل‌ساز است (landmark mismatch)
4. ❌ معمولاً به 20-40px می‌رسد (نه زیر 10px)

### ✅ مزایای Heatmap Approach:

```
Input Image → Model → Heatmaps (192x192) → Extract Coordinates
```

**مزایا:**
1. ✅ Model یک **spatial representation** یاد می‌گیرد
2. ✅ Heatmap loss دقیق‌تر است (Gaussian around ground truth)
3. ✅ **Soft-argmax** برای sub-pixel accuracy
4. ✅ معمولاً به **5-15px** می‌رسد! 🎯

---

## 📊 مقایسه:

| روش | Pixel Error | دقت | سرعت |
|-----|-------------|-----|------|
| **Direct Regression** | 20-40px | متوسط | ⚡⚡⚡ سریع |
| **Heatmap** | **5-15px** | **عالی** | ⚡⚡ متوسط |

---

## 🏗️ Architecture:

### Model Structure:

```
HRNet Backbone (pretrained)
    ↓
Feature Maps (512 channels, 48x48)
    ↓
Upsampling Layers (x4)
    ↓
Heatmaps (2 channels, 192x192)
    ↓
Soft-argmax
    ↓
Coordinates [p1_x, p1_y, p2_x, p2_y]
```

### Heatmap Generation:

```python
# Ground truth heatmap (Gaussian)
heatmap = exp(-((x - x_gt)² + (y - y_gt)²) / (2σ²))

# σ = 3.0 pixels (در heatmap space)
# این یک "ناحیه" به model می‌دهد نه فقط یک نقطه
```

### Coordinate Extraction:

```python
# Soft-argmax (weighted average)
x = Σ(x_i * heatmap[i,j]) / Σ(heatmap[i,j])
y = Σ(y_i * heatmap[i,j]) / Σ(heatmap[i,j])

# این sub-pixel accuracy می‌دهد!
```

---

## 📈 Loss Function:

### Combined Loss:

```python
Total Loss = Heatmap Loss + Coordinate Loss

Heatmap Loss = MSE(pred_heatmap, gt_heatmap)
Coordinate Loss = L1(pred_coords, gt_coords)

Weight: 1.0 * Heatmap + 0.5 * Coordinate
```

**چرا ترکیبی؟**
- ✅ Heatmap loss: Model یاد می‌گیرد "کجا" landmark است
- ✅ Coordinate loss: Model یاد می‌گیرد "دقیقاً کجا" است

---

## ⚙️ Hyperparameters:

### بهینه شده برای < 10px:

```python
image_size = 768        # وضوح بالا
heatmap_size = 192      # 1/4 resolution (کافی برای دقت)
sigma = 3.0             # Gaussian spread
batch_size = 4          # برای RTX 3070 Ti
learning_rate = 0.001   # پایدار
epochs = 200            # با early stopping
```

---

## 🎯 انتظارات:

### با 100 تصویر:

| Metric | انتظار |
|--------|--------|
| **Pixel Error** | **8-15px** ✅ |
| **Best Case** | **3-8px** 🌟 |
| **Worst Case** | 20-30px |
| **Val Loss** | 0.001-0.003 |

### با 200+ تصویر:

| Metric | انتظار |
|--------|--------|
| **Pixel Error** | **5-10px** ✅ |
| **Best Case** | **2-5px** 🌟 |
| **Worst Case** | 15-25px |

---

## 🚀 استفاده:

### Training:

```cmd
cd aariz
train_heatmap.bat
```

یا:

```cmd
python train_p1_p2_heatmap.py
```

### Testing:

```python
from model_heatmap import HRNetP1P2HeatmapDetector
import torch

# Load model
model = HRNetP1P2HeatmapDetector(num_landmarks=2, output_size=192)
checkpoint = torch.load('models/hrnet_p1p2_heatmap_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    heatmaps = model(image_tensor)  # (1, 2, 192, 192)
    coords = model.extract_coordinates(heatmaps)  # (1, 4)
    
    # Denormalize
    p1_x = coords[0, 0] * image_width
    p1_y = coords[0, 1] * image_height
    p2_x = coords[0, 2] * image_width
    p2_y = coords[0, 3] * image_height
```

---

## 📊 Progress Tracking:

### علائم خوب:

```
Epoch 20:
  Pixel Error: 25.3 px  ← در حال کاهش
  Val Loss: 0.0045

Epoch 50:
  Pixel Error: 12.8 px  ← نزدیک به هدف!
  Val Loss: 0.0021

Epoch 100:
  Pixel Error: 8.5 px   ← زیر 10px! 🎉
  Val Loss: 0.0012
```

### اگر هنوز بالاست:

```
Epoch 50:
  Pixel Error: 45.2 px  ← هنوز بالاست

راه‌حل:
1. بررسی کیفیت annotations
2. افزایش heatmap_size به 256
3. کاهش sigma به 2.0
4. بیشتر data (200+ images)
```

---

## 🔧 Tuning برای دقت بیشتر:

### اگر می‌خواهید < 5px:

```python
# در train_p1_p2_heatmap.py:
heatmap_size = 256      # افزایش از 192
sigma = 2.0             # کاهش از 3.0 (دقیق‌تر)
learning_rate = 0.0005  # کاهش برای fine-tuning
coord_weight = 1.0      # افزایش از 0.5
```

**هشدار:** این تنظیمات ممکن است overfitting ایجاد کنند!

---

## 📝 خلاصه:

| ویژگی | Direct Regression | Heatmap |
|-------|-------------------|---------|
| **دقت** | 20-40px | **5-15px** ✅ |
| **پیچیدگی** | ساده | متوسط |
| **سرعت** | سریع | متوسط |
| **Data Augmentation** | مشکل‌ساز | آسان‌تر |
| **Sub-pixel Accuracy** | خیر | **بله** ✅ |

---

## 🎉 نتیجه:

**Heatmap approach برای دقت زیر 10px بهترین انتخاب است!**

**شروع کنید:**

```cmd
cd aariz
train_heatmap.bat
```

**زمان تخمینی:** 3-5 ساعت  
**هدف:** < 10px error ✅













