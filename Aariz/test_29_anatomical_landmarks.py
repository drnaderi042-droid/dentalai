"""
تست 29 لندمارک آناتومیک با مدل اصلی
"""
import torch
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


class AnatomicalLandmarkDataset(Dataset):
    """Dataset برای تست 29 لندمارک آناتومیک"""
    def __init__(self, annotations_dir, images_dir, image_size=768):
        self.image_size = image_size
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        
        # ترتیب لندمارک‌ها
        self.landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        
        annotation_files = list(self.annotations_dir.glob('*.json'))
        
        self.samples = []
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                ceph_id = data.get('ceph_id', '')
                if not ceph_id:
                    continue
                
                # پیدا کردن image
                image_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = self.images_dir / f"{ceph_id}{ext}"
                    if test_path.exists():
                        image_path = test_path
                        break
                
                if not image_path:
                    # جستجوی بیشتر
                    for img_file in self.images_dir.glob('*'):
                        if img_file.stem == ceph_id or (ceph_id in img_file.name and img_file.suffix in ['.png', '.jpg', '.jpeg']):
                            image_path = img_file
                            break
                
                if not image_path:
                    continue
                
                # استخراج landmarks
                landmarks_dict = {}
                if 'landmarks' in data:
                    for lm in data['landmarks']:
                        if 'value' in lm and 'x' in lm['value'] and 'y' in lm['value']:
                            symbol = lm.get('symbol', '')
                            if symbol in self.landmark_symbols:
                                landmarks_dict[symbol] = {
                                    'x': lm['value']['x'],
                                    'y': lm['value']['y']
                                }
                
                # فقط اگر حداقل 25 لندمارک داشته باشد
                if len(landmarks_dict) >= 25:
                    self.samples.append({
                        'image_path': str(image_path),
                        'landmarks': landmarks_dict
                    })
            except Exception as e:
                continue
        
        print(f"Loaded {len(self.samples)} samples with anatomical landmarks")
        
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
        
        # Ground truth coordinates
        gt_coords = np.zeros((29, 2), dtype=np.float32)
        landmarks_dict = sample['landmarks']
        
        for i, symbol in enumerate(self.landmark_symbols):
            if symbol in landmarks_dict:
                lm = landmarks_dict[symbol]
                gt_coords[i, 0] = lm['x'] / original_size[0]
                gt_coords[i, 1] = lm['y'] / original_size[1]
        
        return image_tensor, gt_coords, sample['image_path']


def test_29_anatomical_landmarks(
    model_path='checkpoint_best_768.pth',
    annotations_dir='Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists',
    images_dir='Aariz/train/Cephalograms',
    image_size=768,
    device='cuda',
    max_samples=100
):
    """تست 29 لندمارک آناتومیک"""
    print("="*80)
    print("Testing 29 Anatomical Landmarks")
    print("="*80)
    
    # بارگذاری مدل
    print(f"\n[1/4] Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # تشخیص تعداد لندمارک‌ها - مدل اصلی 29 لندمارک دارد
    num_landmarks = 29  # مدل اصلی checkpoint_best_768.pth همیشه 29 لندمارک دارد
    
    model = HRNetLandmarkModel(num_landmarks=num_landmarks, width=32)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully! ({num_landmarks} landmarks)")
    
    # ایجاد dataset
    print(f"\n[2/4] Loading dataset...")
    dataset = AnatomicalLandmarkDataset(annotations_dir, images_dir, image_size)
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
            
            # پیش‌بینی - استفاده مستقیم از مدل
            heatmaps_pred = model(images)
            heatmaps_pred = torch.sigmoid(heatmaps_pred)
            
            # استخراج مختصات
            coords_pred = extract_coordinates_from_heatmaps(heatmaps_pred, image_size)
            
            # فقط 29 لندمارک اول
            pred_coords = coords_pred[:, :29]
            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords.numpy())
    
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    all_gt_coords = np.concatenate(all_gt_coords, axis=0)
    
    print(f"\n[4/4] Calculating metrics...")
    
    # محاسبه خطاهای پیکسلی
    pixel_errors = calculate_pixel_error(all_pred_coords, all_gt_coords, image_size)
    
    # نتایج
    print(f"\n" + "="*80)
    print("RESULTS - Anatomical Landmarks (1-29)")
    print("="*80)
    
    landmark_symbols = dataset.landmark_symbols
    mre_per_landmark = []
    
    for i in range(29):
        mre = np.mean(pixel_errors[:, i])
        mre_per_landmark.append(mre)
        symbol = landmark_symbols[i]
        status = "OK" if mre < 10 else "WARNING" if mre < 20 else "ERROR"
        print(f"  Landmark {i+1:2d} ({symbol:>4s}): {mre:6.2f} px [{status}]")
    
    mre_overall = np.mean(pixel_errors)
    print(f"\n  Overall Anatomical MRE: {mre_overall:.2f} px")
    
    # SDR
    sdr_5px = []
    sdr_10px = []
    sdr_20px = []
    
    for i in range(29):
        errors_i = pixel_errors[:, i]
        sdr_5px.append(np.sum(errors_i <= 5) / len(errors_i) * 100.0)
        sdr_10px.append(np.sum(errors_i <= 10) / len(errors_i) * 100.0)
        sdr_20px.append(np.sum(errors_i <= 20) / len(errors_i) * 100.0)
    
    print(f"\n  Anatomical SDR:")
    print(f"    SDR@5px:  {np.mean(sdr_5px):.1f}%")
    print(f"    SDR@10px: {np.mean(sdr_10px):.1f}%")
    print(f"    SDR@20px: {np.mean(sdr_20px):.1f}%")
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Total samples tested: {len(dataset)}")
    print(f"  Overall Anatomical MRE: {mre_overall:.2f} px")
    print(f"  Anatomical SDR@10px: {np.mean(sdr_10px):.1f}%")
    print(f"  Anatomical SDR@20px: {np.mean(sdr_20px):.1f}%")
    print("="*80)
    
    return {
        'mre_overall': mre_overall,
        'sdr_10px': np.mean(sdr_10px),
        'sdr_20px': np.mean(sdr_20px)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test 29 anatomical landmarks')
    parser.add_argument('--model', type=str, default='checkpoint_best_768.pth')
    parser.add_argument('--annotations_dir', type=str, 
                        default='Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists')
    parser.add_argument('--images_dir', type=str, default='Aariz/train/Cephalograms')
    parser.add_argument('--max_samples', type=int, default=100)
    
    args = parser.parse_args()
    
    try:
        results = test_29_anatomical_landmarks(
            model_path=args.model,
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            max_samples=args.max_samples
        )
        print("\n[OK] Testing completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()

