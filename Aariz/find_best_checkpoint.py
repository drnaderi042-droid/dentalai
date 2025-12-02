"""
اسکریپت برای پیدا کردن بهترین checkpoint بر اساس MRE
"""

import os
import torch
import json
from pathlib import Path

def load_checkpoint_metrics(checkpoint_path):
    """بارگذاری metrics از checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # بررسی metrics
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            return {
                'mre_mm': metrics.get('mre_mm', None),
                'sdr_2mm': metrics.get('sdr_2mm', None),
                'epoch': checkpoint.get('epoch', None),
            }
        elif 'best_mre' in checkpoint:
            return {
                'mre_mm': checkpoint.get('best_mre', None),
                'sdr_2mm': None,
                'epoch': checkpoint.get('epoch', None),
            }
        else:
            return None
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def find_best_checkpoints(checkpoints_dir='checkpoints'):
    """پیدا کردن بهترین checkpoint ها"""
    checkpoints_dir = Path(checkpoints_dir)
    
    if not checkpoints_dir.exists():
        print(f"Directory {checkpoints_dir} not found!")
        return
    
    # پیدا کردن تمام checkpoint files
    checkpoint_files = list(checkpoints_dir.glob('checkpoint_*.pth'))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoints_dir}")
        return
    
    print("="*80)
    print("🔍 بررسی Checkpoint ها برای پیدا کردن بهترین مدل")
    print("="*80)
    print(f"\n📂 پوشه: {checkpoints_dir}")
    print(f"📊 تعداد checkpoint ها: {len(checkpoint_files)}\n")
    
    results = []
    
    for ckpt_path in checkpoint_files:
        metrics = load_checkpoint_metrics(ckpt_path)
        if metrics and metrics['mre_mm'] is not None:
            results.append({
                'path': ckpt_path,
                'name': ckpt_path.name,
                'mre_mm': metrics['mre_mm'],
                'sdr_2mm': metrics['sdr_2mm'],
                'epoch': metrics['epoch'],
            })
    
    if not results:
        print("❌ هیچ checkpoint با metrics یافت نشد!")
        return
    
    # مرتب کردن بر اساس MRE (کمتر = بهتر)
    results.sort(key=lambda x: x['mre_mm'])
    
    print("="*80)
    print("📊 نتایج (مرتب شده بر اساس MRE - کمتر بهتر است)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Epoch':<8} {'MRE (mm)':<12} {'SDR @ 2mm':<15} {'File':<40}")
    print("-"*80)
    
    for i, result in enumerate(results[:20], 1):  # نمایش 20 تا اول
        sdr_str = f"{result['sdr_2mm']:.2f}%" if result['sdr_2mm'] else "N/A"
        epoch_str = str(result['epoch']) if result['epoch'] is not None else "N/A"
        print(f"{i:<6} {epoch_str:<8} {result['mre_mm']:<12.4f} {sdr_str:<15} {result['name']:<40}")
    
    print("\n" + "="*80)
    print("🏆 بهترین Checkpoint:")
    print("="*80)
    best = results[0]
    print(f"\n✅ بهترین: {best['name']}")
    print(f"   Epoch: {best['epoch']}")
    print(f"   MRE: {best['mre_mm']:.4f} mm")
    if best['sdr_2mm']:
        print(f"   SDR @ 2mm: {best['sdr_2mm']:.2f}%")
    print(f"   Path: {best['path']}")
    
    # مقایسه با checkpoint_best.pth
    best_path = checkpoints_dir / 'checkpoint_best.pth'
    if best_path.exists():
        best_metrics = load_checkpoint_metrics(best_path)
        if best_metrics and best_metrics['mre_mm']:
            print(f"\n📌 checkpoint_best.pth فعلی:")
            print(f"   MRE: {best_metrics['mre_mm']:.4f} mm")
            if best_metrics['mre_mm'] > best['mre_mm']:
                print(f"\n⚠️  هشدار: checkpoint_best.pth بهترین نیست!")
                print(f"   بهتر است از {best['name']} استفاده کنید")
                print(f"\n💡 پیشنهاد:")
                print(f"   # در app_aariz.py یا inference.py")
                print(f"   CHECKPOINT_PATH = '{best['path']}'")
            else:
                print(f"\n✅ checkpoint_best.pth بهترین است!")
    
    # ذخیره نتایج
    output_file = checkpoints_dir / 'checkpoint_ranking.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 نتایج در {output_file} ذخیره شد")
    
    return results

if __name__ == '__main__':
    import sys
    
    checkpoints_dir = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints'
    find_best_checkpoints(checkpoints_dir)

