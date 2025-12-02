"""
اسکریپت سریع برای بهبود مدل با fine-tuning
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Quick model improvement with fine-tuning')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pth',
                       help='Path to checkpoint to fine-tune')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (lower than initial training)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 Quick Model Improvement - Fine-tuning")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print("\n" + "="*80)
    
    # بررسی وجود checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint not found: {args.checkpoint}")
        print("\n💡 پیشنهاد:")
        print("   1. ابتدا بهترین checkpoint را پیدا کنید:")
        print("      python find_best_checkpoint.py")
        print("   2. سپس از این اسکریپت استفاده کنید")
        return
    
    # ساخت دستور train_optimized
    command = f"""python train_optimized.py \\
  --model hrnet \\
  --resume {args.checkpoint} \\
  --epochs {args.epochs} \\
  --learning_rate {args.lr} \\
  --batch_size {args.batch_size} \\
  --image_size 512 512 \\
  --mixed_precision \\
  --use_ema \\
  --gradient_accumulation_steps 2 \\
  --warmup_epochs 5"""
    
    print("\n📝 دستور پیشنهادی:")
    print("-"*80)
    print(command)
    print("-"*80)
    
    print("\n✅ برای اجرا، دستور بالا را کپی و اجرا کنید")
    print("   یا از این اسکریپت با --execute استفاده کنید (در حال توسعه)")

if __name__ == '__main__':
    main()

