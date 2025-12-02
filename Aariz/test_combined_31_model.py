"""
اسکریپت تست مدل ترکیبی 31 لندمارک
محاسبه MRE و SDR برای همه 31 لندمارک
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel
from utils import heatmap_to_coordinates, calculate_mre
import torchvision.transforms as transforms


def extract_coordinates_from_heatmaps(heatmaps, image_size):
    """
    استخراج مختصات از heatmap‌ها
    Args:
        heatmaps: (batch, num_landmarks, H, W) tensor
        image_size: اندازه تصویر
    Returns:
        coords: (batch, num_landmarks, 2) numpy array - مختصات normalized [0, 1]
    """
    batch_size, num_landmarks, H, W = heatmaps.shape
    
    # استفاده از soft-argmax برای استخراج مختصات
    y_coords = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, 1, W)
    
    # Normalize to [0, 1]
    y_coords = y_coords / (H - 1) if H > 1 else y_coords
    x_coords = x_coords / (W - 1) if W > 1 else x_coords
    
    # Weighted average
    heatmaps_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
    
    x_mean = (heatmaps * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    y_mean = (heatmaps * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    
    # Stack: (batch, num_landmarks, 2)
    coords = torch.stack([x_mean, y_mean], dim=-1)
    
    return coords.detach().cpu().numpy()


def calculate_pixel_error(pred_coords, gt_coords, image_size):
    """
    محاسبه خطای پیکسلی
    Args:
        pred_coords: (batch, num_landmarks, 2) - مختصات پیش‌بینی شده normalized
        gt_coords: (batch, num_landmarks, 2) - مختصات ground truth normalized
        image_size: اندازه تصویر
    Returns:
        errors: (batch, num_landmarks) - خطاهای پیکسلی
    """
    # Convert to pixel coordinates
    pred_px = pred_coords * image_size  # (batch, num_landmarks, 2)
    gt_px = gt_coords * image_size  # (batch, num_landmarks, 2)
    
    # Calculate radial error for each landmark
    errors = np.sqrt(np.sum((pred_px - gt_px) ** 2, axis=2))  # (batch, num_landmarks)
    
    return errors


def calculate_sdr_pixel(errors, thresholds=[5, 10, 20]):
    """
    محاسبه SDR در پیکسل
    Args:
        errors: (batch, num_landmarks) - خطاهای پیکسلی
        thresholds: لیست threshold ها در پیکسل
    Returns:
        sdr_dict: دیکشنری با SDR برای هر threshold
    """
    sdr_dict = {}
    total = errors.size
    
    for threshold in thresholds:
        success = np.sum(errors <= threshold)
        sdr = (success / total) * 100.0
        sdr_dict[f'sdr_{threshold}px'] = sdr
    
    return sdr_dict


class TestDataset:
    """Dataset برای تست مدل - استفاده از همان dataset که در fine-tuning استفاده شده"""
    def __init__(self, annotations_file, images_dir, image_size=768):
        self.image_size = image_size
        self.images_dir = Path(images_dir)
        
        # بارگذاری annotations از فایل JSON (همان فایلی که در fine-tuning استفاده شده)
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
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
                                    'x': value.get('x', 0) / 100.0,  # Normalize to [0, 1]
                                    'y': value.get('y', 0) / 100.0
                                }
                            elif 'p2' in labels:
                                p2 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                    
                    if p1 and p2:
                        # پیدا کردن image path
                        image_path = None
                        if 'data' in item and 'image' in item['data']:
                            image_path = item['data']['image'].replace('/data/upload/', '')
                        elif 'file_upload' in item:
                            image_path = item['file_upload']
                        
                        if image_path:
                            self.samples.append({
                                'image_path': image_path,
                                'p1': p1,
                                'p2': p2
                            })
        
        print(f"Loaded {len(self.samples)} samples with P1/P2 annotations")
        
        # Transform - استفاده از normalization مشابه training
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # بارگذاری تصویر
        image_path = self.images_dir / sample['image_path']
        if not image_path.exists():
            # Try alternative path
            image_path = self.images_dir / os.path.basename(sample['image_path'])
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Preprocess
        image_tensor = self.transform(image)
        
        # Ground truth coordinates (normalized [0, 1])
        # فقط P1/P2 را داریم، بقیه را صفر می‌گذاریم (برای تست فقط P1/P2 مهم است)
        gt_coords = np.zeros((31, 2), dtype=np.float32)
        
        # Fill P1/P2 (indices 29, 30) - از annotation
        gt_coords[29, 0] = sample['p1']['x']
        gt_coords[29, 1] = sample['p1']['y']
        gt_coords[30, 0] = sample['p2']['x']
        gt_coords[30, 1] = sample['p2']['y']
        
        # برای anatomical landmarks (1-29)، از مدل اصلی استفاده می‌کنیم
        # اما در تست فعلی فقط P1/P2 را ارزیابی می‌کنیم
        
        return image_tensor, gt_coords, sample['image_path']


def test_model(
    model_path='checkpoint_best_768_combined_31_finetuned.pth',
    annotations_file='annotations_p1_p2.json',
    images_dir='Aariz/train/Cephalograms',
    image_size=768,
    device='cuda',
    max_samples=None
):
    """
    تست مدل روی dataset
    """
    print("="*80)
    print("Testing Combined 31-Landmark Model")
    print("="*80)
    
    # بارگذاری مدل
    print(f"\n[1/4] Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Trying original combined model...")
        model_path = 'checkpoint_best_768_combined_31.pth'
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model = HRNetLandmarkModel(num_landmarks=31, width=32)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully!")
    
    # ایجاد dataset
    print(f"\n[2/4] Loading dataset...")
    dataset = TestDataset(annotations_file, images_dir, image_size)
    if max_samples:
        dataset.samples = dataset.samples[:max_samples]
    print(f"  Test samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("  [ERROR] No samples found! Please check annotation file.")
        return
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # تست
    print(f"\n[3/4] Testing model...")
    all_pred_coords = []
    all_gt_coords = []
    
    with torch.no_grad():
        for images, gt_coords, image_paths in tqdm(dataloader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            
            # پیش‌بینی
            heatmaps_pred = model(images)
            heatmaps_pred = torch.sigmoid(heatmaps_pred)
            
            # استخراج مختصات
            coords_pred = extract_coordinates_from_heatmaps(heatmaps_pred, image_size)
            
            # ذخیره
            all_pred_coords.append(coords_pred)
            all_gt_coords.append(gt_coords.numpy())
    
    # ترکیب همه batch ها
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)  # (N, 31, 2)
    all_gt_coords = np.concatenate(all_gt_coords, axis=0)  # (N, 31, 2)
    
    print(f"\n[4/4] Calculating metrics...")
    
    # محاسبه خطاهای پیکسلی برای همه 31 لندمارک
    pixel_errors_all = calculate_pixel_error(all_pred_coords, all_gt_coords, image_size)  # (N, 31)
    
    # نتایج برای 29 لندمارک آناتومیک
    anatomical_errors = pixel_errors_all[:, :29]
    mre_anatomical = np.mean(anatomical_errors)
    
    # SDR برای anatomical
    sdr_5px_anatomical = []
    sdr_10px_anatomical = []
    sdr_20px_anatomical = []
    
    for i in range(29):
        errors_i = anatomical_errors[:, i]
        sdr_5px_anatomical.append(np.sum(errors_i <= 5) / len(errors_i) * 100.0)
        sdr_10px_anatomical.append(np.sum(errors_i <= 10) / len(errors_i) * 100.0)
        sdr_20px_anatomical.append(np.sum(errors_i <= 20) / len(errors_i) * 100.0)
    
    print(f"\n" + "="*80)
    print("RESULTS - Anatomical Landmarks (1-29)")
    print("="*80)
    print(f"  Overall Anatomical MRE: {mre_anatomical:.2f} px")
    print(f"  Anatomical SDR:")
    print(f"    SDR@5px:  {np.mean(sdr_5px_anatomical):.1f}%")
    print(f"    SDR@10px: {np.mean(sdr_10px_anatomical):.1f}%")
    print(f"    SDR@20px: {np.mean(sdr_20px_anatomical):.1f}%")
    
    # نتایج برای P1/P2
    pixel_errors_p1p2 = pixel_errors_all[:, 29:31]
    mre_p1 = np.mean(pixel_errors_p1p2[:, 0])
    mre_p2 = np.mean(pixel_errors_p1p2[:, 1])
    mre_p1p2_avg = np.mean(pixel_errors_p1p2)
    
    sdr_p1 = calculate_sdr_pixel(pixel_errors_p1p2[:, 0:1], thresholds=[5, 10, 20])
    sdr_p2 = calculate_sdr_pixel(pixel_errors_p1p2[:, 1:2], thresholds=[5, 10, 20])
    sdr_avg = calculate_sdr_pixel(pixel_errors_p1p2, thresholds=[5, 10, 20])
    
    print(f"\n" + "="*80)
    print("RESULTS - P1/P2 Landmarks (30-31)")
    print("="*80)
    print(f"  MRE - P1: {mre_p1:.2f} px | P2: {mre_p2:.2f} px | Avg: {mre_p1p2_avg:.2f} px")
    print(f"  SDR (5px)  - P1: {sdr_p1['sdr_5px']:.1f}% | P2: {sdr_p2['sdr_5px']:.1f}% | Avg: {sdr_avg['sdr_5px']:.1f}%")
    print(f"  SDR (10px) - P1: {sdr_p1['sdr_10px']:.1f}% | P2: {sdr_p2['sdr_10px']:.1f}% | Avg: {sdr_avg['sdr_10px']:.1f}%")
    print(f"  SDR (20px) - P1: {sdr_p1['sdr_20px']:.1f}% | P2: {sdr_p2['sdr_20px']:.1f}% | Avg: {sdr_avg['sdr_20px']:.1f}%")
    
    # خلاصه کلی
    print(f"\n" + "="*80)
    print("SUMMARY - All 31 Landmarks")
    print("="*80)
    print(f"  Total samples tested: {len(dataset)}")
    print(f"  Anatomical MRE (1-29): {mre_anatomical:.2f} px")
    print(f"  P1/P2 MRE (30-31): {mre_p1p2_avg:.2f} px")
    print(f"  Overall MRE: {np.mean(pixel_errors_all):.2f} px")
    print(f"\n  Anatomical SDR@10px: {np.mean(sdr_10px_anatomical):.1f}%")
    print(f"  P1/P2 SDR@10px: {sdr_avg['sdr_10px']:.1f}%")
    print(f"  Overall SDR@10px: {np.mean([np.mean(sdr_10px_anatomical), sdr_avg['sdr_10px']]):.1f}%")
    print("="*80)
    
    return {
        'mre_anatomical': mre_anatomical,
        'mre_p1p2': mre_p1p2_avg,
        'mre_overall': np.mean(pixel_errors_all),
        'sdr_10px_anatomical': np.mean(sdr_10px_anatomical),
        'sdr_10px_p1p2': sdr_avg['sdr_10px']
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test combined 31-landmark model')
    parser.add_argument('--model', type=str, default='checkpoint_best_768_combined_31_finetuned.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--annotations_file', type=str, 
                        default='annotations_p1_p2.json',
                        help='Path to annotations JSON file (same as used in fine-tuning)')
    parser.add_argument('--images_dir', type=str, default='Aariz/train/Cephalograms',
                        help='Path to images directory')
    parser.add_argument('--image_size', type=int, default=768,
                        help='Image size for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (None = all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    try:
        results = test_model(
            model_path=args.model,
            annotations_file=args.annotations_file,
            images_dir=args.images_dir,
            image_size=args.image_size,
            device=args.device,
            max_samples=args.max_samples
        )
        print("\n[OK] Testing completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

