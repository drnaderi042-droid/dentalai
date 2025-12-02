"""
اسکریپت کمک برای انتخاب بهترین استراتژی بهبود
"""

import os
import json

print("="*80)
print("🎯 راهنمای انتخاب استراتژی بهبود")
print("="*80)

print("\n📊 وضعیت فعلی:")
print("   - MRE: 1.99 mm ✅")
print("   - SDR @ 2mm: 65.52%")
print("   - هدف: MRE ~1.7mm, SDR @ 2mm ~72%")
print("   - فاصله تا هدف: 6.48% (13 لندمارک بیشتر نیاز است)")

print("\n" + "="*80)
print("🎯 استراتژی‌های پیشنهادی:")
print("="*80)

strategies = [
    {
        "name": "Fine-tuning ملایم",
        "description": "سریع‌ترین و بی‌خطرترین روش",
        "time": "2-3 ساعت",
        "epochs": 30,
        "lr": "1e-5",
        "expected_improvement": "SDR → 68-70%",
        "recommendation": "⭐⭐⭐⭐⭐",
        "command": """python train2.py \\
    --resume checkpoints/checkpoint_best.pth \\
    --model hrnet \\
    --image_size 256 256 \\
    --batch_size 16 \\
    --lr 1e-5 \\
    --warmup_epochs 2 \\
    --epochs 30 \\
    --loss adaptive_wing \\
    --mixed_precision"""
    },
    {
        "name": "Fine-tuning متوسط",
        "description": "تعادل بین زمان و بهبود",
        "time": "4-5 ساعت",
        "epochs": 50,
        "lr": "5e-5",
        "expected_improvement": "SDR → 70-72%",
        "recommendation": "⭐⭐⭐⭐",
        "command": """python train2.py \\
    --resume checkpoints/checkpoint_best.pth \\
    --model hrnet \\
    --image_size 256 256 \\
    --batch_size 16 \\
    --lr 5e-5 \\
    --warmup_epochs 3 \\
    --epochs 50 \\
    --loss adaptive_wing \\
    --mixed_precision"""
    },
    {
        "name": "آموزش از اول",
        "description": "بیشترین بهبود احتمالی اما زمان‌بر",
        "time": "8-12 ساعت",
        "epochs": 100,
        "lr": "5e-4",
        "expected_improvement": "SDR → 72-75%",
        "recommendation": "⭐⭐⭐",
        "command": """python train2.py \\
    --dataset_path Aariz \\
    --model hrnet \\
    --image_size 256 256 \\
    --batch_size 16 \\
    --lr 5e-4 \\
    --warmup_epochs 5 \\
    --epochs 100 \\
    --loss adaptive_wing \\
    --mixed_precision"""
    }
]

for i, strategy in enumerate(strategies, 1):
    print(f"\n{i}. {strategy['name']} {strategy['recommendation']}")
    print(f"   توضیح: {strategy['description']}")
    print(f"   زمان: {strategy['time']}")
    print(f"   Epochs: {strategy['epochs']}")
    print(f"   Learning Rate: {strategy['lr']}")
    print(f"   بهبود مورد انتظار: {strategy['expected_improvement']}")

print("\n" + "="*80)
print("💡 پیشنهاد:")
print("="*80)
print("\n✅ شروع با Fine-tuning ملایم (گزینه 1)")
print("   - سریع و بی‌خطر")
print("   - احتمال موفقیت 70-80%")
print("   - اگر جواب نداد، به گزینه 2 بروید")

print("\n" + "="*80)
print("📝 دستورات:")
print("="*80)

print("\nبرای Fine-tuning ملایم:")
print("-" * 80)
print("""
# روش 1: استفاده از batch file
finetune_model.bat

# روش 2: دستی
python train2.py \\
    --resume checkpoints/checkpoint_best.pth \\
    --model hrnet \\
    --image_size 256 256 \\
    --batch_size 16 \\
    --lr 1e-5 \\
    --warmup_epochs 2 \\
    --epochs 30 \\
    --loss adaptive_wing \\
    --mixed_precision
""")

print("\n" + "="*80)
print("⚠️  نکات مهم:")
print("="*80)
print("\n1. قبل از شروع، backup بگیرید:")
print("   copy checkpoints\\checkpoint_best.pth checkpoints\\checkpoint_best_backup.pth")
print("\n2. در طول آموزش، Tensorboard را باز کنید:")
print("   tensorboard --logdir logs")
print("\n3. اگر Validation MRE افزایش یافت، متوقف کنید (Ctrl+C)")
print("\n4. بعد از آموزش، تست کنید:")
print("   python compare_new_results.py")

print("\n" + "="*80)
print("📚 مستندات کامل:")
print("="*80)
print("\n- IMPROVEMENT_STRATEGY.md: راهنمای جامع")
print("- FINETUNE_GUIDE_FA.md: راهنمای سریع Fine-tuning")
print("- FIXED_IMAGE_SIZE_ISSUE.md: مشکل رفع شده (256×256)")

print("\n" + "="*80)

