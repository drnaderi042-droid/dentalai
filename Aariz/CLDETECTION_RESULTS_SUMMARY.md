# نتایج تست دقت مدل CLdetection2023 روی دیتاست Aariz

## ⚠️ وضعیت فعلی

نصب کامل MMPose نیاز به تنظیمات خاص دارد (CUDA_HOME و محیط conda). برای تست کامل، باید مراحل زیر را انجام دهید:

## 📋 مراحل نصب و تست کامل

### 1. ایجاد محیط Conda
```bash
conda create -n LMD python=3.10
conda activate LMD
```

### 2. نصب Dependencies
```bash
cd CLdetection2023
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
cd mmpose_package/mmpose
pip install -e .
```

### 3. اجرای تست
```bash
cd ../../Aariz
python test_cldetection_final.py
```

## 📊 اطلاعات مدل CLdetection2023

### مشخصات مدل:
- **Architecture**: SRPose (Super-Resolution Pose)
- **Backbone**: HRNet-W48
- **Input Size**: 1024x1024
- **Heatmap Size**: 1024x1024
- **تعداد لندمارک‌ها**: 19 لندمارک سفالومتری

### لندمارک‌های مدل:
1. S - Sella
2. N - Nasion
3. Or - Orbitale
4. A - Point A
5. B - Point B
6. PNS - Posterior Nasal Spine
7. ANS - Anterior Nasal Spine
8. U1 - Upper Incisor Tip
9. L1 - Lower Incisor Tip
10. Me - Menton
11. U6 - Upper Molar Tip
12. L6 - Lower Molar Tip
13. Go - Gonion
14. Pog - Pogonion
15. Gn - Gnathion
16. Ar - Articulare
17. Co - Condylion
18. Po - Porion
19. R - Ramus point

## 🔍 لندمارک‌های مشترک با Aariz (15 عدد)

از 19 لندمارک مدل CLdetection2023، **15 عدد** با دیتاست Aariz مشترک است:

| # | لندمارک | توضیحات |
|---|---------|---------|
| 1 | S | Sella |
| 2 | N | Nasion |
| 3 | Or | Orbitale |
| 4 | A | Point A (Subspinale) |
| 5 | B | Point B (Supramentale) |
| 6 | PNS | Posterior Nasal Spine |
| 7 | ANS | Anterior Nasal Spine |
| 8 | Me | Menton |
| 9 | Go | Gonion |
| 10 | Pog | Pogonion |
| 11 | Gn | Gnathion |
| 12 | Ar | Articulare |
| 13 | Co | Condylion |
| 14 | Po | Porion |
| 15 | R | Ramus point |

### لندمارک‌های فقط در CLdetection2023 (4 عدد):
- U1, L1, U6, L6

### لندمارک‌های فقط در Aariz (14 عدد):
- LIA, LIT, LMT, LPM, Li, Ls, N`, Pn, Pog`, Sn, UIA, UIT, UMT, UPM

## 📈 Metrics مورد ارزیابی

پس از اجرای تست، این metrics محاسبه می‌شود:

1. **Mean Radial Error (MRE)**: میانگین خطا در میلی‌متر
2. **Median Error**: میانه خطا
3. **Standard Deviation**: انحراف معیار
4. **Success Detection Rate (SDR)**: درصد موفقیت در آستانه‌های:
   - SDR @ 1mm
   - SDR @ 2mm
   - SDR @ 2.5mm
   - SDR @ 3mm
   - SDR @ 4mm
5. **Per-landmark Statistics**: آمار برای هر لندمارک به صورت جداگانه

## 📝 فایل‌های ایجاد شده

1. `test_cldetection_final.py` - اسکریپت تست کامل
2. `test_cldetection_batch.py` - اسکریپت تست دسته‌ای
3. `test_cldetection_accuracy.py` - اسکریپت با راهنمای کامل
4. `CLDETECTION_TEST_GUIDE.md` - راهنمای کامل تست

## 🔗 منابع

- Repository: https://github.com/5k5000/CLdetection2023
- Paper: https://arxiv.org/pdf/2309.17143.pdf
- Challenge: MICCAI CLdetection2023

## ⚡ نکات مهم

1. **Mapping لندمارک‌ها**: مدل CLdetection2023 خروجی 19 لندمارک دارد که باید به 15 لندمارک مشترک با Aariz نگاشت شوند.

2. **Scale کردن مختصات**: مدل روی resolution 1024x1024 آموزش دیده است. باید مختصات را به اندازه اصلی تصویر scale کنید.

3. **Pixel Size**: برای تبدیل خطا از پیکسل به میلی‌متر، از `pixel_size` از فایل CSV استفاده کنید.

4. **نتایج**: پس از اجرای موفق تست، نتایج در فایل `cldetection_accuracy_results.json` ذخیره می‌شود.
















