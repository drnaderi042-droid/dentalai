# راهنمای سریع اجرای دستورات

## ⚠️ مهم: در Windows

در Windows PowerShell و CMD، syntax با Linux/Mac فرق دارد!

## ✅ روش‌های مختلف برای Windows

### روش 1: یک خط کامل (ساده‌ترین) ⭐

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --lr 2e-4 --warmup_epochs 3 --loss adaptive_wing --epochs 100
```

**کپی و paste کنید!** ✅

### روش 2: چند خطی در PowerShell

در PowerShell از **backtick (`)** استفاده کنید (نه backslash):

```powershell
python train.py `
    --resume checkpoints/checkpoint_best.pth `
    --dataset_path Aariz `
    --model resnet `
    --lr 2e-4 `
    --warmup_epochs 3 `
    --loss adaptive_wing `
    --epochs 100
```

**نکته:** بعد از هر ` باید Enter بزنید، PowerShell خودش ادامه می‌دهد.

### روش 3: چند خطی در CMD

در Command Prompt از `^` استفاده کنید:

```cmd
python train.py ^
    --resume checkpoints/checkpoint_best.pth ^
    --dataset_path Aariz ^
    --model resnet ^
    --lr 2e-4 ^
    --warmup_epochs 3 ^
    --loss adaptive_wing ^
    --epochs 100
```

## 📋 مثال‌های آماده برای Copy-Paste

### Fine-tuning از بهترین checkpoint:

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --lr 1e-4 --warmup_epochs 3 --loss adaptive_wing --epochs 50
```

### شروع جدید:

```powershell
python train.py --dataset_path Aariz --model resnet --lr 5e-4 --warmup_epochs 5 --loss adaptive_wing --epochs 100
```

### تغییر فقط LR:

```powershell
python train.py --resume checkpoints/checkpoint_latest.pth --dataset_path Aariz --model resnet --lr 2e-4 --warmup_epochs 5 --loss adaptive_wing --epochs 100
```

### تغییر فقط Warmup:

```powershell
python train.py --resume checkpoints/checkpoint_latest.pth --dataset_path Aariz --model resnet --lr 5e-4 --warmup_epochs 10 --loss adaptive_wing --epochs 100
```

## 🔍 تفاوت PowerShell و CMD

| Terminal | کاراکتر ادامه خط | مثال |
|----------|------------------|------|
| **PowerShell** | `` ` `` (backtick) | `python train.py ` |
| **CMD** | `^` (caret) | `python train.py ^` |
| **Linux/Mac** | `\` (backslash) | `python train.py \` |

## 💡 نکات

1. **ساده‌ترین:** همه را در یک خط بنویسید (روش 1)
2. **قابل خواندن:** در PowerShell از backtick استفاده کنید (روش 2)
3. **نکته مهم:** در PowerShell، `#` برای کامنت است، اما در command line نمی‌توانید کامنت بگذارید!

## ⚠️ خطاهای رایج

### خطا: `--lr 2e-4 \`
```
--lr: unrecognized arguments: \
```

**راه حل:** از `\` استفاده نکنید! در PowerShell از `` ` `` استفاده کنید یا همه را در یک خط بنویسید.

### خطا: `# LR جدید`
```
--lr: unrecognized arguments: #
```

**راه حل:** کامنت `#` را حذف کنید! در command line نمی‌توانید کامنت بگذارید.

---

**پیشنهاد: از روش 1 (یک خط) استفاده کنید! ساده‌ترین است. ✅**

