# خلاصه مشکل و راهکار

## 🔍 مشکل

مدل Hugging Face ([cwlachap/hrnet-cephalometric-landmark-detection](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection)) با dataset ISBI train شده و روی dataset Aariz شما عملکرد ضعیفی دارد:

- **MRE روی ISBI**: ~1.2-1.6mm ✅
- **MRE روی Aariz**: ~47mm ❌

## ✅ راهکار: Fine-tuning

Fine-tuning مدل Hugging Face با dataset Aariz برای بهبود عملکرد.

## 🚀 اجرای Fine-tuning

### روش 1: استفاده از Batch Script (ساده‌تر)

```bash
cd Aariz
.\finetune_huggingface_model.bat
```

### روش 2: دستی

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

## 📊 انتظارات

بعد از fine-tuning:
- **MRE**: باید به زیر 5mm برسد (یا بهتر)
- **SDR @ 2mm**: باید بالای 50% باشد
- **مختصات**: باید در محدوده صحیح باشند

## ⚠️ نکات مهم

1. **Learning Rate پایین**: از `1e-5` استفاده کنید (نه `5e-4`) برای fine-tuning
2. **Epochs**: 50 epoch معمولاً کافی است
3. **Monitoring**: از TensorBoard برای monitoring استفاده کنید
4. **Checkpoints**: بهترین مدل در `checkpoints/checkpoint_best.pth` ذخیره می‌شود

## 📝 بعد از Fine-tuning

بعد از اتمام fine-tuning، می‌توانید مدل جدید را تست کنید:

```bash
# کپی مدل fine-tuned به cephx_service
copy checkpoints\checkpoint_best.pth ..\cephx_service\model\hrnet_cephalometric_finetuned.pth

# تست
python test_hrnet_python_frontend_comparison.py --mode all
```
















