"""
اسکریپت تست کامل مدل 31 لندمارک - شامل 29 لندمارک آناتومیک + 2 P1/P2
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel
import torchvision.transforms as transforms

# ترتیب لندمارک‌های آناتومیک (29 لندمارک) - مطابق با inference.py
ANATOMICAL_LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]


def extract_coordinates_from_heatmaps(heatmaps, image_size):
    """استخراج مختصات از heatmap‌ها"""
    batch_size, num_landmarks, H, W = heatmaps.shape
    
    y_coords = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, 1, W)
    
    y_coords = y_coords / (H - 1) if H > 1 else y_coords
    x_coords = x_coords / (W - 1) if W > 1 else x_coords
    
    heatmaps_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
    
    x_mean = (heatmaps * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    y_mean = (heatmaps * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    
    coords = torch.stack([x_mean, y_mean], dim=-1)
    return coords.detach().cpu().numpy()


def calculate_pixel_error(pred_coords, gt_coords, image_size):
    """محاسبه خطای پیکسلی"""
    pred_px = pred_coords * image_size
    gt_px = gt_coords * image_size
    errors = np.sqrt(np.sum((pred_px - gt_px) ** 2, axis=2))
    return errors


def calculate_sdr_pixel(errors, thresholds=[5, 10, 20]):
    """محاسبه SDR در پیکسل"""
    sdr_dict = {}
    total = errors.size
    
    for threshold in thresholds:
        success = np.sum(errors <= threshold)
        sdr = (success / total) * 100.0
        sdr_dict[f'sdr_{threshold}px'] = sdr
    
    return sdr_dict


class Full31LandmarkDataset(Dataset):
    """Dataset برای تست کامل 31 لندمارک"""
    def __init__(self, annotations_dir, images_dir, p1p2_annotations_file, image_size=768):
        self.image_size = image_size
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        
        # بارگذاری P1/P2 annotations
        with open(p1p2_annotations_file, 'r', encoding='utf-8') as f:
            p1p2_data = json.load(f)
        
        # ایجاد mapping از image path به P1/P2
        self.p1p2_map = {}
        for item in p1p2_data:
            if 'annotations' in item and item['annotations']:
                annotation = item['annotations'][0]
                if 'result' in annotation:
                    p1 = None
                    p2 = None
                    
                    for result_item in annotation['result']:
                        if result_item.get('type') == 'keypointlabels':
                            value = result_item.get('value', {})
                            labels = value.get('keypointlabels', [])
                            
                            if 'p1' in labels:
                                p1 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                            elif 'p2' in labels:
                                p2 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                    
                    if p1 and p2:
                        image_url = item['data'].get('image', '')
                        image_filename = image_url.split('/')[-1]
                        if '-' in image_filename:
                            parts = image_filename.split('-')
                            if len(parts[0]) == 8:
                                image_filename = '-'.join(parts[1:])
                        self.p1p2_map[image_filename] = {'p1': p1, 'p2': p2}
        
        # بارگذاری annotation‌های آناتومیک
        annotation_files = list(self.annotations_dir.glob('*.json'))
        
        self.samples = []
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                ceph_id = data.get('ceph_id', '')
                if not ceph_id:
                    continue
                
                # پیدا کردن image - چند روش مختلف
                image_path = None
                
                # روش 1: مستقیم با ceph_id
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = self.images_dir / f"{ceph_id}{ext}"
                    if test_path.exists():
                        image_path = test_path
                        break
                
                # روش 2: جستجو در همه فایل‌های تصویر
                if not image_path:
                    for img_file in self.images_dir.glob('*'):
                        if img_file.stem == ceph_id or img_file.name.startswith(ceph_id):
                            image_path = img_file
                            break
                
                # روش 3: جستجو با UUID prefix (اگر وجود دارد)
                if not image_path:
                    for img_file in self.images_dir.glob('*'):
                        if img_file.name.endswith(f"{ceph_id}.png") or img_file.name.endswith(f"{ceph_id}.jpg"):
                            image_path = img_file
                            break
                        # اگر UUID prefix دارد
                        if '-' in img_file.name:
                            parts = img_file.stem.split('-')
                            if len(parts) > 1 and parts[-1] == ceph_id:
                                image_path = img_file
                                break
                
                if not image_path:
                    continue
                
                # استخراج landmarks آناتومیک
                landmarks_dict = {}
                if 'landmarks' in data:
                    for lm in data['landmarks']:
                        if 'value' in lm and 'x' in lm['value'] and 'y' in lm['value']:
                            symbol = lm.get('symbol', '')
                            landmarks_dict[symbol] = {
                                'x': lm['value']['x'],
                                'y': lm['value']['y']
                            }
                
                # پیدا کردن P1/P2 - چند روش مختلف
                p1p2_data = None
                image_filename = image_path.name
                
                # روش 1: مستقیم با filename
                p1p2_data = self.p1p2_map.get(image_filename)
                
                # روش 2: بدون extension
                if not p1p2_data:
                    p1p2_data = self.p1p2_map.get(image_path.stem)
                
                # روش 3: با UUID prefix
                if not p1p2_data and '-' in image_filename:
                    parts = image_filename.split('-')
                    if len(parts) > 1:
                        # بدون UUID prefix
                        alt_name = '-'.join(parts[1:])
                        p1p2_data = self.p1p2_map.get(alt_name)
                        # یا فقط آخرین part
                        if not p1p2_data:
                            p1p2_data = self.p1p2_map.get(parts[-1])
                
                # روش 4: جستجو در همه keys
                if not p1p2_data:
                    for key in self.p1p2_map.keys():
                        if ceph_id in key or key in image_filename or image_filename in key:
                            p1p2_data = self.p1p2_map[key]
                            break
                
                if p1p2_data and len(landmarks_dict) >= 20:  # حداقل 20 لندمارک آناتومیک
                    self.samples.append({
                        'image_path': str(image_path),
                        'landmarks': landmarks_dict,
                        'p1': p1p2_data['p1'],
                        'p2': p1p2_data['p2']
                    })
            except Exception as e:
                continue
        
        print(f"Loaded {len(self.samples)} samples with full 31 landmarks")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size  # (width, height)
        
        image_tensor = self.transform(image)
        
        # Ground truth coordinates (31 landmarks)
        gt_coords = np.zeros((31, 2), dtype=np.float32)
        
        # Fill anatomical landmarks (0-28)
        landmarks_dict = sample['landmarks']
        for i, symbol in enumerate(ANATOMICAL_LANDMARK_SYMBOLS):
            if symbol in landmarks_dict:
                lm = landmarks_dict[symbol]
                gt_coords[i, 0] = lm['x'] / original_size[0]
                gt_coords[i, 1] = lm['y'] / original_size[1]
        
        # Fill P1/P2 (29, 30)
        gt_coords[29, 0] = sample['p1']['x']
        gt_coords[29, 1] = sample['p1']['y']
        gt_coords[30, 0] = sample['p2']['x']
        gt_coords[30, 1] = sample['p2']['y']
        
        return image_tensor, gt_coords, sample['image_path']


def test_all_landmarks(
    model_path='checkpoint_best_768_combined_31_finetuned.pth',
    annotations_dir='Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists',
    images_dir='Aariz/train/Cephalograms',
    p1p2_annotations_file='annotations_p1_p2.json',
    image_size=768,
    device='cuda',
    max_samples=None
):
    """تست کامل 31 لندمارک"""
    print("="*80)
    print("Testing All 31 Landmarks (29 Anatomical + 2 P1/P2)")
    print("="*80)
    
    # بارگذاری مدل
    print(f"\n[1/4] Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model = HRNetLandmarkModel(num_landmarks=31, width=32)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully!")
    
    # ایجاد dataset
    print(f"\n[2/4] Loading dataset...")
    dataset = Full31LandmarkDataset(annotations_dir, images_dir, p1p2_annotations_file, image_size)
    if max_samples:
        dataset.samples = dataset.samples[:max_samples]
    print(f"  Test samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("  [ERROR] No samples found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # تست
    print(f"\n[3/4] Testing model...")
    all_pred_coords = []
    all_gt_coords = []
    
    with torch.no_grad():
        for images, gt_coords, image_paths in tqdm(dataloader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            
            heatmaps_pred = model(images)
            heatmaps_pred = torch.sigmoid(heatmaps_pred)
            
            coords_pred = extract_coordinates_from_heatmaps(heatmaps_pred, image_size)
            
            all_pred_coords.append(coords_pred)
            all_gt_coords.append(gt_coords.numpy())
    
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    all_gt_coords = np.concatenate(all_gt_coords, axis=0)
    
    print(f"\n[4/4] Calculating metrics...")
    
    # محاسبه خطاهای پیکسلی
    pixel_errors = calculate_pixel_error(all_pred_coords, all_gt_coords, image_size)
    
    # نتایج برای 29 لندمارک آناتومیک
    print(f"\n" + "="*80)
    print("RESULTS - Anatomical Landmarks (1-29)")
    print("="*80)
    
    anatomical_errors = pixel_errors[:, :29]
    mre_per_landmark = []
    
    for i in range(29):
        mre = np.mean(anatomical_errors[:, i])
        mre_per_landmark.append(mre)
        symbol = ANATOMICAL_LANDMARK_SYMBOLS[i]
        status = "OK" if mre < 10 else "WARNING" if mre < 20 else "ERROR"
        print(f"  Landmark {i+1:2d} ({symbol:>4s}): {mre:6.2f} px [{status}]")
    
    mre_anatomical = np.mean(anatomical_errors)
    print(f"\n  Overall Anatomical MRE: {mre_anatomical:.2f} px")
    
    # SDR برای anatomical
    sdr_5px_anatomical = []
    sdr_10px_anatomical = []
    sdr_20px_anatomical = []
    
    for i in range(29):
        errors_i = anatomical_errors[:, i]
        sdr_5px_anatomical.append(np.sum(errors_i <= 5) / len(errors_i) * 100.0)
        sdr_10px_anatomical.append(np.sum(errors_i <= 10) / len(errors_i) * 100.0)
        sdr_20px_anatomical.append(np.sum(errors_i <= 20) / len(errors_i) * 100.0)
    
    print(f"\n  Anatomical SDR:")
    print(f"    SDR@5px:  {np.mean(sdr_5px_anatomical):.1f}%")
    print(f"    SDR@10px: {np.mean(sdr_10px_anatomical):.1f}%")
    print(f"    SDR@20px: {np.mean(sdr_20px_anatomical):.1f}%")
    
    # نتایج برای P1/P2
    print(f"\n" + "="*80)
    print("RESULTS - P1/P2 Landmarks (30-31)")
    print("="*80)
    
    p1p2_errors = pixel_errors[:, 29:31]
    mre_p1 = np.mean(p1p2_errors[:, 0])
    mre_p2 = np.mean(p1p2_errors[:, 1])
    mre_p1p2 = np.mean(p1p2_errors)
    
    print(f"  MRE - P1: {mre_p1:.2f} px | P2: {mre_p2:.2f} px | Avg: {mre_p1p2:.2f} px")
    
    sdr_p1 = calculate_sdr_pixel(p1p2_errors[:, 0:1], thresholds=[5, 10, 20])
    sdr_p2 = calculate_sdr_pixel(p1p2_errors[:, 1:2], thresholds=[5, 10, 20])
    sdr_avg_p1p2 = calculate_sdr_pixel(p1p2_errors, thresholds=[5, 10, 20])
    
    print(f"  SDR (5px)  - P1: {sdr_p1['sdr_5px']:.1f}% | P2: {sdr_p2['sdr_5px']:.1f}% | Avg: {sdr_avg_p1p2['sdr_5px']:.1f}%")
    print(f"  SDR (10px) - P1: {sdr_p1['sdr_10px']:.1f}% | P2: {sdr_p2['sdr_10px']:.1f}% | Avg: {sdr_avg_p1p2['sdr_10px']:.1f}%")
    print(f"  SDR (20px) - P1: {sdr_p1['sdr_20px']:.1f}% | P2: {sdr_p2['sdr_20px']:.1f}% | Avg: {sdr_avg_p1p2['sdr_20px']:.1f}%")
    
    # خلاصه کلی
    print(f"\n" + "="*80)
    print("SUMMARY - All 31 Landmarks")
    print("="*80)
    print(f"  Total samples tested: {len(dataset)}")
    print(f"  Anatomical MRE (1-29): {mre_anatomical:.2f} px")
    print(f"  P1/P2 MRE (30-31): {mre_p1p2:.2f} px")
    print(f"  Overall MRE: {np.mean(pixel_errors):.2f} px")
    print(f"\n  Anatomical SDR@10px: {np.mean(sdr_10px_anatomical):.1f}%")
    print(f"  P1/P2 SDR@10px: {sdr_avg_p1p2['sdr_10px']:.1f}%")
    print(f"  Overall SDR@10px: {np.mean([np.mean(sdr_10px_anatomical), sdr_avg_p1p2['sdr_10px']]):.1f}%")
    print("="*80)
    
    return {
        'mre_anatomical': mre_anatomical,
        'mre_p1p2': mre_p1p2,
        'mre_overall': np.mean(pixel_errors),
        'sdr_10px_anatomical': np.mean(sdr_10px_anatomical),
        'sdr_10px_p1p2': sdr_avg_p1p2['sdr_10px']
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test all 31 landmarks')
    parser.add_argument('--model', type=str, default='checkpoint_best_768_combined_31_finetuned.pth')
    parser.add_argument('--annotations_dir', type=str, 
                        default='Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists')
    parser.add_argument('--images_dir', type=str, default='Aariz/train/Cephalograms')
    parser.add_argument('--p1p2_file', type=str, default='annotations_p1_p2.json')
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    try:
        results = test_all_landmarks(
            model_path=args.model,
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            p1p2_annotations_file=args.p1p2_file,
            max_samples=args.max_samples
        )
        print("\n[OK] Testing completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()

