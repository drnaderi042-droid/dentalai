"""بررسی وضعیت training"""
import os
import torch
import glob

aariz_path = "Aariz"
checkpoints_dir = os.path.join(aariz_path, "checkpoints")

# بررسی checkpoint جدید
new_checkpoint = os.path.join(checkpoints_dir, "checkpoint_best.pth")
old_checkpoint = os.path.join(aariz_path, "checkpoint_best_768.pth")

print("="*80)
print("بررسی وضعیت Training")
print("="*80)

if os.path.exists(new_checkpoint):
    checkpoint = torch.load(new_checkpoint, map_location='cpu')
    epoch = checkpoint.get('epoch', 'N/A')
    best_mre = checkpoint.get('best_mre', 'N/A')
    val_metrics = checkpoint.get('val_metrics', {})
    
    print(f"\nCheckpoint جديد (checkpoints/checkpoint_best.pth):")
    print(f"  Epoch: {epoch}")
    print(f"  Best MRE: {best_mre}")
    if val_metrics:
        print(f"  SDR @ 2mm: {val_metrics.get('sdr_2mm', 'N/A')}")
        print(f"  Validation MRE: {val_metrics.get('mre_mm', 'N/A')}")
    
    # بررسی زمان ایجاد
    import datetime
    mod_time = os.path.getmtime(new_checkpoint)
    print(f"  زمان ایجاد: {datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("\n[WARNING] Checkpoint جديد يافت نشد!")

if os.path.exists(old_checkpoint):
    checkpoint_old = torch.load(old_checkpoint, map_location='cpu')
    epoch_old = checkpoint_old.get('epoch', 'N/A')
    best_mre_old = checkpoint_old.get('best_mre', 'N/A')
    
    print(f"\nCheckpoint قديمي (checkpoint_best_768.pth):")
    print(f"  Epoch: {epoch_old}")
    print(f"  Best MRE: {best_mre_old}")
    
    mod_time_old = os.path.getmtime(old_checkpoint)
    import datetime
    print(f"  زمان ایجاد: {datetime.datetime.fromtimestamp(mod_time_old).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if os.path.exists(new_checkpoint):
        if epoch > epoch_old:
            print(f"\n✅ Training انجام شده: از epoch {epoch_old} به {epoch} رسيده")
        else:
            print(f"\n⚠️ Training جديد انجام نشده: epoch تغيير نکرده ({epoch_old} -> {epoch})")
            print("   احتمالاً فقط resume شده و epoch جديدي train نشده")

print("\n" + "="*80)















