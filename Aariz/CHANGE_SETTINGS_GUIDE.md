# راهنمای تغییر تنظیمات در حین آموزش

## ⚠️ محدودیت مهم

**اگر آموزش در حال اجراست:**
- ❌ نمی‌توانید فایل `train.py` را تغییر دهید
- ❌ نمی‌توانید تنظیمات را مستقیماً تغییر دهید
- ✅ **باید آموزش را متوقف کنید و از checkpoint resume کنید**

## ✅ راه حل: Resume از Checkpoint

### مرحله 1: متوقف کردن آموزش فعلی

در terminal که آموزش در حال اجراست:
```
Ctrl + C
```

### مرحله 2: Resume با تنظیمات جدید

```bash
# مثال: تغییر LR و warmup
python train.py \
    --resume checkpoints/checkpoint_best.pth \
    --dataset_path Aariz \
    --model resnet \
    --lr 2e-4 \              # LR جدید
    --warmup_epochs 3 \      # Warmup جدید
    --loss adaptive_wing \
    --epochs 100
```

### مرحله 3: راه‌های مختلف تغییر تنظیمات

#### تغییر فقط LR:
```bash
python train.py \
    --resume checkpoints/checkpoint_latest.pth \
    --lr 1e-4 \
    --warmup_epochs 5
```

#### تغییر فقط Warmup:
```bash
python train.py \
    --resume checkpoints/checkpoint_latest.pth \
    --lr 5e-4 \
    --warmup_epochs 10  # افزایش warmup
```

#### Fine-tuning با LR پایین:
```bash
python train.py \
    --resume checkpoints/checkpoint_best.pth \
    --lr 1e-5 \          # LR خیلی پایین
    --warmup_epochs 2 \
    --epochs 50
```

## 📝 مثال‌های کاربردی

### مثال 1: کاهش LR برای Fine-tuning

```bash
# از checkpoint بهترین مدل
python train.py \
    --resume checkpoints/checkpoint_best.pth \
    --dataset_path Aariz \
    --model resnet \
    --lr 1e-4 \              # کاهش LR از 5e-4
    --warmup_epochs 3 \      # کاهش warmup
    --loss adaptive_wing \
    --batch_size 8 \
    --epochs 50
```

### مثال 2: افزایش LR اگر مدل یاد نمی‌گیرد

```bash
python train.py \
    --resume checkpoints/checkpoint_latest.pth \
    --lr 1e-3 \              # افزایش LR
    --warmup_epochs 5 \
    --loss adaptive_wing \
    --epochs 50
```

### مثال 3: تغییر Warmup برای سازگاری با LR جدید

```bash
# اگر LR را زیاد کردید، warmup را هم افزایش دهید
python train.py \
    --resume checkpoints/checkpoint_latest.pth \
    --lr 8e-4 \
    --warmup_epochs 8 \      # warmup بیشتر برای LR بالاتر
    --loss adaptive_wing \
    --epochs 50
```

## 🔧 تنظیمات پیشنهادی بر اساس وضعیت

| وضعیت | LR | Warmup | توضیح |
|-------|----|--------|-------|
| **شروع از اول** | 5e-4 | 5 | پیش‌فرض جدید |
| **Fine-tuning (MRE < 10mm)** | 1e-4 | 3 | LR پایین |
| **Fine-tuning (MRE < 5mm)** | 5e-5 | 2 | LR خیلی پایین |
| **Stuck (MRE گیر کرده)** | 1e-3 | 8 | LR بالاتر |
| **Overfitting** | 2e-4 | 3 | LR متوسط |

## 💡 نکات مهم

1. **Checkpoint را انتخاب کنید:**
   - `checkpoint_best.pth`: بهترین مدل
   - `checkpoint_latest.pth`: آخرین epoch

2. **LR را هوشمندانه تغییر دهید:**
   - اگر MRE در حال بهبود است: LR را کاهش دهید
   - اگر MRE گیر کرده: LR را افزایش دهید

3. **Warmup را تنظیم کنید:**
   - LR بالاتر = Warmup بیشتر
   - Fine-tuning = Warmup کمتر

4. **تعداد Epochs:**
   - Fine-tuning: 30-50 epoch
   - از اول: 100 epoch

## ⚡ تغییر سریع (بدون Resume)

اگر فقط می‌خواهید LR را در یک epoch خاص تغییر دهید:

```python
# در training loop، بعد از validation:
if epoch == 20:  # در epoch 20
    for pg in optimizer.param_groups:
        pg['lr'] = 1e-4  # LR جدید
    print(f"LR changed to {optimizer.param_groups[0]['lr']}")
```

اما این نیاز به تغییر کد دارد و باید آموزش را restart کنید.

---

**توصیه: همیشه از checkpoint resume کنید! ✅**

