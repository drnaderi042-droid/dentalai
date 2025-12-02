# تحلیل مشکل مدل Hugging Face روی Dataset Aariz

## 🔍 وضعیت فعلی

### مدل Hugging Face
- **منبع**: [cwlachap/hrnet-cephalometric-landmark-detection](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection)
- **Dataset Training**: ISBI Lateral Cephalograms
- **Performance**: MRE ~1.2-1.6mm روی ISBI dataset
- **Input Size**: 768×768 pixels

### Dataset Aariz
- **Dataset شما**: Aariz
- **Image Size**: 1968 × 2225 (متفاوت از ISBI)
- **Aspect Ratio**: 0.8845 (متفاوت از ISBI: 0.806)

### مشکل
- **MRE فعلی**: 47.06mm ❌ (باید زیر 2mm باشد)
- **علت**: مدل با dataset متفاوتی train شده

## 📊 تحلیل تفاوت‌ها

از نتایج تشخیصی:
- **Pred X range**: 945-1100 (فقط در مرکز)
- **GT X range**: 291-1585 (در کل تصویر)
- **Offset سیستماتیک**: میانگین Diff X: -103.92px, Diff Y: -67.57px

این نشان می‌دهد که:
1. مدل برای distribution متفاوتی train شده
2. ممکن است image preprocessing متفاوت باشد
3. نیاز به fine-tuning یا adaptation دارد

## ✅ راهکارها

### گزینه 1: Fine-tuning مدل Hugging Face (پیشنهادی)

Fine-tune مدل Hugging Face با dataset Aariz:

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

**مزایا:**
- استفاده از pretrained weights
- سریع‌تر از train از ابتدا
- معمولاً نتایج بهتری دارد

### گزینه 2: بررسی Preprocessing مدل Hugging Face

ممکن است مدل Hugging Face preprocessing متفاوتی داشته باشد. بررسی کنید:
- Normalization (ImageNet vs custom)
- Image augmentation
- Heatmap generation

### گزینه 3: Domain Adaptation

استفاده از تکنیک‌های domain adaptation برای تطبیق مدل با dataset جدید.

## 🔧 بررسی Preprocessing

از [Hugging Face model card](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection):
- Input Size: 768×768
- Dataset: ISBI Lateral Cephalograms
- Performance: MRE ~1.2-1.6mm

باید بررسی کنید که:
1. آیا normalization یکسان است؟
2. آیا image size processing یکسان است؟
3. آیا heatmap generation یکسان است؟

## 📝 خلاصه

**مشکل**: مدل Hugging Face با ISBI train شده، شما روی Aariz تست می‌کنید

**راهکار اصلی**: Fine-tuning با dataset Aariz

**انتظارات بعد از Fine-tuning**:
- MRE: زیر 2mm
- SDR @ 2mm: بالای 70%

## 🚀 گام بعدی

1. Fine-tune مدل Hugging Face با dataset Aariz
2. یا استفاده از مدل خودتان که قبلاً با Aariz train کرده‌اید

