# رفع مشکل MRE بالا

## مشکل شناسایی شده

- Loss کاهش می‌یابد اما MRE بهبود نمی‌یابد (~51mm)
- این نشان می‌دهد مدل heatmap‌ها را به درستی یاد نگرفته

## علل احتمالی

1. **Image size 128×128 خیلی کوچک است**: 
   - HRNet و ResNet برای image size بزرگ‌تر طراحی شده‌اند
   - برای 128×128، جزئیات مهم از دست می‌رود

2. **Learning rate 1e-3 خیلی بالا است**:
   - برای image size کوچک، LR باید کمتر باشد
   - LR بالا باعث overshoot و عدم convergence می‌شود

3. **Batch size 40 خیلی بزرگ است**:
   - برای 128×128، batch size بزرگ gradient ها را ناپایدار می‌کند
   - بهتر است batch size را کاهش دهیم

## راه حل

### تنظیمات بهینه برای HRNet:

```powershell
python train2.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 16 --epochs 100
```

### یا اگر VRAM کافی ندارید:

```powershell
python train2.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 8 --epochs 100
```

## تغییرات اعمال شده

1. ✅ Image size: 128×128 → 256×256
2. ✅ Learning rate: 1e-3 → 5e-4
3. ✅ Batch size: 40 → 16 (یا 8)
4. ✅ Model: HRNet (بهترین برای landmark detection)

## انتظارات

با این تنظیمات:
- بعد از 5-10 epoch باید MRE به زیر 20mm برسد
- بعد از 20-30 epoch باید MRE به زیر 10mm برسد
- بعد از 50+ epoch باید MRE به زیر 5mm برسد (هدف)

