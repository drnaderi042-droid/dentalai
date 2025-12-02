# راهنمای تست دقت مدل CLdetection2023

## 📋 خلاصه

این راهنما نحوه تست دقت مدل CLdetection2023 روی دیتاست Aariz را توضیح می‌دهد.

## 🔍 لندمارک‌های مشترک (15 عدد)

مدل CLdetection2023 **19 لندمارک** را تشخیص می‌دهد که **15 عدد** آن با دیتاست Aariz مشترک است:

1. **S** - Sella
2. **N** - Nasion
3. **Or** - Orbitale
4. **A** - Point A
5. **B** - Point B
6. **PNS** - Posterior Nasal Spine
7. **ANS** - Anterior Nasal Spine
8. **Me** - Menton
9. **Go** - Gonion
10. **Pog** - Pogonion
11. **Gn** - Gnathion
12. **Ar** - Articulare
13. **Co** - Condylion
14. **Po** - Porion
15. **R** - Ramus point

## 🚀 روش اجرای تست

### روش 1: استفاده از Repository اصلی (توصیه می‌شود)

```bash
# 1. Clone repository
cd ..
git clone https://github.com/5k5000/CLdetection2023.git
cd CLdetection2023

# 2. نصب dependencies
conda create -n LMD python=3.10
conda activate LMD
pip install -r requirements.txt
pip install -U openmim
cd mmpose_package/mmpose
pip install -e .
mim install mmengine
mim install "mmcv>=2.0.0"
cd ../..

# 3. کپی مدل
copy ..\Aariz\model_pretrained_on_train_and_val.pth .

# 4. تست روی یک تصویر
python inference_single_image.py ^
    --config "configs/CLdetection2023/srpose_s2.py" ^
    --checkpoint "model_pretrained_on_train_and_val.pth" ^
    --image_path "../Aariz/test/Cephalograms/cks2ip8fp29yl0yuf6ry9266i.png"
```

### روش 2: استفاده از اسکریپت خودکار

اگر MMPose نصب شده باشد:

```bash
cd Aariz
python test_cldetection_batch.py
```

## 📊 Metrics مورد ارزیابی

- **Mean Radial Error (MRE)**: میانگین خطا در میلی‌متر
- **Median Error**: میانه خطا
- **Standard Deviation**: انحراف معیار
- **Success Detection Rate (SDR)**: درصد موفقیت در آستانه‌های مختلف
- **Per-landmark Statistics**: آمار برای هر لندمارک

## 📝 فایل‌های ایجاد شده

- `test_cldetection_accuracy.py`: اسکریپت تست با راهنمای کامل
- `test_cldetection_batch.py`: اسکریپت تست دسته‌ای (نیاز به MMPose)
- `test_cldetection_accuracy.bat`: فایل batch برای اجرای تست

## 🔗 منابع

- Repository: https://github.com/5k5000/CLdetection2023
- Paper: https://arxiv.org/pdf/2309.17143.pdf
















