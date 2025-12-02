"""
اسکریپت برای مانیتورینگ progress training
"""

import os
import glob
import time
from datetime import datetime

def monitor_training():
    """مانیتورینگ training progress"""
    logs_dir = "logs"
    checkpoints_dir = "checkpoints"
    
    print("="*80)
    print("Monitoring Training Progress")
    print("="*80)
    
    # پیدا کردن آخرین log directory
    if os.path.exists(logs_dir):
        log_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
        if log_dirs:
            latest_log = max(log_dirs, key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)))
            log_path = os.path.join(logs_dir, latest_log)
            print(f"\nLatest log directory: {latest_log}")
            
            # بررسی وجود event files
            event_files = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
            if event_files:
                latest_event = max(event_files, key=os.path.getmtime)
                mod_time = os.path.getmtime(latest_event)
                print(f"Latest event file: {os.path.basename(latest_event)}")
                print(f"Last modified: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # بررسی اینکه آیا اخیراً update شده
                time_diff = time.time() - mod_time
                if time_diff < 60:
                    print(f"Status: [ACTIVE] Training in progress (updated {time_diff:.0f}s ago)")
                elif time_diff < 300:
                    print(f"Status: [RECENT] Last update {time_diff/60:.1f} minutes ago")
                else:
                    print(f"Status: [IDLE] Last update {time_diff/60:.1f} minutes ago")
    
    # بررسی checkpoint ها
    if os.path.exists(checkpoints_dir):
        checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint_*.pth"))
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            print(f"\nLatest checkpoints:")
            for i, ckpt in enumerate(checkpoints[:5]):
                mod_time = os.path.getmtime(ckpt)
                size_mb = os.path.getsize(ckpt) / (1024 * 1024)
                print(f"  {i+1}. {os.path.basename(ckpt)}")
                print(f"     Size: {size_mb:.1f} MB")
                print(f"     Modified: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("برای مشاهده جزئیات بیشتر، TensorBoard را اجرا کنید:")
    print(f"  tensorboard --logdir={logs_dir}")
    print("="*80)

if __name__ == "__main__":
    monitor_training()















