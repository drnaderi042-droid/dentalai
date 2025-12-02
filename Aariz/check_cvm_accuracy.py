"""
Check CVM model accuracy from checkpoints
"""
import torch
import os

checkpoint_dir = "CVM/checkpoints"

# Check all epoch checkpoints
epochs_to_check = [0, 5, 10, 15, 20, 25, 30, 35]

print("=" * 70)
print("CVM Model Training Progress")
print("=" * 70)
print(f"{'Epoch':<8} | {'Val Accuracy':<12} | {'Val Loss':<10} | {'Best Accuracy':<12} | {'Stage Accuracies'}")
print("-" * 70)

for ep in epochs_to_check:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{ep}.pth")
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            val_metrics = ckpt.get('val_metrics', {})
            acc = val_metrics.get('accuracy', 0)
            loss = val_metrics.get('loss', 0)
            best = ckpt.get('best_accuracy', 0)
            per_class = val_metrics.get('per_class_accuracy', {})
            
            # Format per-class accuracies
            stage_accs = ", ".join([f"{k}:{v:.1f}%" for k, v in sorted(per_class.items())])
            
            print(f"{ep:<8} | {acc:>10.2f}% | {loss:>9.4f} | {best:>11.2f}% | {stage_accs}")
        except Exception as e:
            print(f"Error loading epoch {ep}: {e}")

# Check best checkpoint
print("\n" + "=" * 70)
print("Best Checkpoint Details")
print("=" * 70)
best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
if os.path.exists(best_path):
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Best Accuracy: {ckpt.get('best_accuracy', 'N/A'):.2f}%")
    val_metrics = ckpt.get('val_metrics', {})
    print(f"Validation Loss: {val_metrics.get('loss', 'N/A'):.4f}")
    print(f"Validation Accuracy: {val_metrics.get('accuracy', 'N/A'):.2f}%")
    print("\nPer-class Accuracy:")
    per_class = val_metrics.get('per_class_accuracy', {})
    for stage, acc in sorted(per_class.items()):
        print(f"  {stage}: {acc:.2f}%")
    
    print("\nTraining Arguments:")
    args = ckpt.get('args', {})
    print(f"  Model: {args.get('model', 'N/A')}")
    print(f"  Image Size: {args.get('image_size', 'N/A')}")
    print(f"  Batch Size: {args.get('batch_size', 'N/A')}")
    print(f"  Learning Rate: {args.get('lr', 'N/A')}")
    print(f"  Epochs: {args.get('epochs', 'N/A')}")
    print(f"  Mixed Precision: {args.get('mixed_precision', 'N/A')}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("Model trained with: train_cvm.py")
print("Script used: train_cvm_768.bat (for 768x768 images)")
print("\nWARNING: Model accuracy is very low (44% best, 34% at epoch 35)")
print("   The model appears to be overfitting or needs better hyperparameters.")

