# راهنمای Fine-tuning مدل Hugging Face با Dataset Aariz

## 🔍 وضعیت فعلی

### مدل Hugging Face
- **منبع**: [cwlachap/hrnet-cephalometric-landmark-detection](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection)
- **Dataset Training**: ISBI Lateral Cephalograms
- **Performance روی ISBI**: MRE ~1.2-1.6mm ✅
- **Performance روی Aariz**: MRE ~47mm ❌

### مشکل
مدل با dataset متفاوتی (ISBI) train شده و روی dataset شما (Aariz) عملکرد ضعیفی دارد.

## ✅ راهکار: Fine-tuning

Fine-tuning مدل Hugging Face با dataset Aariz برای بهبود عملکرد.

## 🚀 مراحل Fine-tuning

### مرحله 1: بررسی ساختار Checkpoint

```bash
cd cephx_service
python -c "import torch; ckpt = torch.load('model/hrnet_cephalometric.pth', map_location='cpu'); print('Keys:', list(ckpt.keys())); print('Config:', ckpt.get('config', {}).get('INPUT', {}))"
```

### مرحله 2: Fine-tuning با Learning Rate پایین

```bash
cd Aariz
python train.py \
  --model hrnet \
  --resume ../cephx_service/model/hrnet_cephalometric.pth \
  --dataset_path Aariz \
  --image_size 768 768 \
  --batch_size 4 \
  --lr 1e-5 \
  --epochs 50 \
  --mixed_precision \
  --loss adaptive_wing
```

**پارامترهای مهم:**
- `--resume`: استفاده از checkpoint Hugging Face
- `--lr 1e-5`: Learning rate پایین برای fine-tuning
- `--epochs 50`: تعداد epochs (می‌توانید کمتر هم بگذارید)
- `--mixed_precision`: برای سرعت بیشتر

### مرحله 3: Fine-tuning با Learning Rate بالاتر (اگر نیاز بود)

اگر بعد از 50 epoch هنوز نتایج خوب نیست:

```bash
python train.py \
  --model hrnet \
  --resume checkpoints/checkpoint_latest.pth \
  --dataset_path Aariz \
  --image_size 768 768 \
  --batch_size 4 \
  --lr 5e-5 \
  --epochs 30 \
  --mixed_precision \
  --loss adaptive_wing
```

## 📊 انتظارات

بعد از fine-tuning:
- **MRE**: باید به زیر 5mm برسد (یا حتی بهتر)
- **SDR @ 2mm**: باید بالای 50% باشد
- **مختصات**: باید در محدوده صحیح باشند

## 🔧 نکات مهم

### 1. Freeze کردن لایه‌های اولیه (اختیاری)

اگر می‌خواهید فقط لایه‌های آخر را fine-tune کنید:

```python
# در train.py
for name, param in model.named_parameters():
    if 'stage4' not in name:  # فقط stage4 را train کنید
        param.requires_grad = False
```

### 2. استفاده از Different Learning Rates

```python
# Learning rate متفاوت برای لایه‌های مختلف
optimizer = optim.AdamW([
    {'params': model.stage4.parameters(), 'lr': 1e-5},
    {'params': model.final_layer.parameters(), 'lr': 5e-5}
])
```

### 3. Monitoring

```bash
# استفاده از TensorBoard
tensorboard --logdir logs
```

## 📝 خلاصه

**مشکل**: مدل Hugging Face با ISBI train شده، روی Aariz عملکرد ضعیفی دارد

**راهکار**: Fine-tuning با dataset Aariz

**مراحل**:
1. Fine-tune با LR پایین (1e-5) برای 50 epochs
2. اگر نیاز بود، ادامه با LR بالاتر
3. تست و ارزیابی

**انتظارات**: MRE زیر 5mm، SDR @ 2mm بالای 50%
















