"""
مقایسه مدل combined (31 لندمارک) با مدل اصلی (29 لندمارک)
برای 29 لندمارک آناتومیک از مدل اصلی استفاده می‌کنیم
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


class P1P2Dataset(Dataset):
    """Dataset برای تست - فقط P1/P2 annotation"""
    def __init__(self, annotations_file, images_dir, image_size=768):
        self.image_size = image_size
        self.images_dir = Path(images_dir)
        
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
                        
                        image_path = self.images_dir / image_filename
                        if not image_path.exists():
                            alt_filename = image_url.split('/')[-1]
                            alt_path = self.images_dir / alt_filename
                            if alt_path.exists():
                                image_path = alt_path
                            else:
                                continue
                        
                        self.samples.append({
                            'image_path': str(image_path),
                            'p1': p1,
                            'p2': p2
                        })
        
        print(f"Loaded {len(self.samples)} samples with P1/P2 annotations")
        
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
        image_tensor = self.transform(image)
        
        # فقط P1/P2 coordinates
        coords = np.zeros((31, 2), dtype=np.float32)
        coords[29, 0] = sample['p1']['x']
        coords[29, 1] = sample['p1']['y']
        coords[30, 0] = sample['p2']['x']
        coords[30, 1] = sample['p2']['y']
        
        return image_tensor, coords, sample['image_path']


def test_models_comparison(
    original_model_path='checkpoint_best_768.pth',
    combined_model_path='checkpoint_best_768_combined_31_finetuned.pth',
    annotations_file='annotations_p1_p2.json',
    images_dir='Aariz/train/Cephalograms',
    image_size=768,
    device='cuda',
    max_samples=100
):
    """مقایسه مدل اصلی و combined"""
    print("="*80)
    print("Comparing Original Model (29 landmarks) vs Combined Model (31 landmarks)")
    print("="*80)
    
    # بارگذاری مدل اصلی (29 لندمارک)
    print(f"\n[1/5] Loading original model (29 landmarks)...")
    orig_checkpoint = torch.load(original_model_path, map_location='cpu', weights_only=False)
    orig_state_dict = orig_checkpoint.get('model_state_dict', orig_checkpoint)
    orig_model = HRNetLandmarkModel(num_landmarks=29, width=32)
    orig_model.load_state_dict(orig_state_dict, strict=False)
    orig_model = orig_model.to(device)
    orig_model.eval()
    print(f"  Original model loaded!")
    
    # بارگذاری مدل combined (31 لندمارک)
    print(f"\n[2/5] Loading combined model (31 landmarks)...")
    comb_checkpoint = torch.load(combined_model_path, map_location='cpu', weights_only=False)
    comb_state_dict = comb_checkpoint.get('model_state_dict', comb_checkpoint)
    comb_model = HRNetLandmarkModel(num_landmarks=31, width=32)
    comb_model.load_state_dict(comb_state_dict, strict=False)
    comb_model = comb_model.to(device)
    comb_model.eval()
    print(f"  Combined model loaded!")
    
    # ایجاد dataset
    print(f"\n[3/5] Loading dataset...")
    dataset = P1P2Dataset(annotations_file, images_dir, image_size)
    if max_samples:
        dataset.samples = dataset.samples[:max_samples]
    print(f"  Test samples: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # تست
    print(f"\n[4/5] Testing models...")
    orig_pred_29 = []
    comb_pred_29 = []
    comb_pred_p1p2 = []
    gt_p1p2 = []
    
    with torch.no_grad():
        for images, gt_coords, image_paths in tqdm(dataloader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            
            # پیش‌بینی با مدل اصلی (29 لندمارک)
            orig_heatmaps = orig_model(images)
            orig_heatmaps = torch.sigmoid(orig_heatmaps)
            orig_coords_29 = extract_coordinates_from_heatmaps(orig_heatmaps, image_size)
            orig_pred_29.append(orig_coords_29)
            
            # پیش‌بینی با مدل combined (31 لندمارک)
            comb_heatmaps = comb_model(images)
            comb_heatmaps = torch.sigmoid(comb_heatmaps)
            comb_coords_31 = extract_coordinates_from_heatmaps(comb_heatmaps, image_size)
            
            # جدا کردن 29 لندمارک آناتومیک و 2 لندمارک P1/P2
            comb_pred_29.append(comb_coords_31[:, :29])
            comb_pred_p1p2.append(comb_coords_31[:, 29:31])
            gt_p1p2.append(gt_coords[:, 29:31].numpy())
    
    orig_pred_29 = np.concatenate(orig_pred_29, axis=0)
    comb_pred_29 = np.concatenate(comb_pred_29, axis=0)
    comb_pred_p1p2 = np.concatenate(comb_pred_p1p2, axis=0)
    gt_p1p2 = np.concatenate(gt_p1p2, axis=0)
    
    print(f"\n[5/5] Calculating metrics...")
    
    # مقایسه 29 لندمارک آناتومیک بین دو مدل
    diff_29 = np.sqrt(np.sum((orig_pred_29 - comb_pred_29) ** 2, axis=2)) * image_size
    avg_diff_29 = np.mean(diff_29)
    max_diff_29 = np.max(diff_29)
    
    print(f"\n" + "="*80)
    print("COMPARISON - 29 Anatomical Landmarks")
    print("="*80)
    print(f"  Average difference between models: {avg_diff_29:.2f} px")
    print(f"  Maximum difference: {max_diff_29:.2f} px")
    print(f"  Note: This shows how similar the 29 landmarks are between")
    print(f"        original model and combined model (should be close to 0)")
    
    # نتایج P1/P2 از مدل combined
    p1p2_errors = calculate_pixel_error(comb_pred_p1p2, gt_p1p2, image_size)
    mre_p1 = np.mean(p1p2_errors[:, 0])
    mre_p2 = np.mean(p1p2_errors[:, 1])
    mre_p1p2 = np.mean(p1p2_errors)
    
    sdr_p1 = calculate_sdr_pixel(p1p2_errors[:, 0:1], thresholds=[5, 10, 20])
    sdr_p2 = calculate_sdr_pixel(p1p2_errors[:, 1:2], thresholds=[5, 10, 20])
    sdr_avg = calculate_sdr_pixel(p1p2_errors, thresholds=[5, 10, 20])
    
    print(f"\n" + "="*80)
    print("RESULTS - P1/P2 from Combined Model (30-31)")
    print("="*80)
    print(f"  MRE - P1: {mre_p1:.2f} px | P2: {mre_p2:.2f} px | Avg: {mre_p1p2:.2f} px")
    print(f"  SDR (5px)  - P1: {sdr_p1['sdr_5px']:.1f}% | P2: {sdr_p2['sdr_5px']:.1f}% | Avg: {sdr_avg['sdr_5px']:.1f}%")
    print(f"  SDR (10px) - P1: {sdr_p1['sdr_10px']:.1f}% | P2: {sdr_p2['sdr_10px']:.1f}% | Avg: {sdr_avg['sdr_10px']:.1f}%")
    print(f"  SDR (20px) - P1: {sdr_p1['sdr_20px']:.1f}% | P2: {sdr_p2['sdr_20px']:.1f}% | Avg: {sdr_avg['sdr_20px']:.1f}%")
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Total samples tested: {len(dataset)}")
    print(f"  29 Landmarks - Avg difference between models: {avg_diff_29:.2f} px")
    print(f"  P1/P2 MRE: {mre_p1p2:.2f} px")
    print(f"  P1/P2 SDR@10px: {sdr_avg['sdr_10px']:.1f}%")
    print(f"  P1/P2 SDR@20px: {sdr_avg['sdr_20px']:.1f}%")
    print("="*80)
    
    return {
        'avg_diff_29': avg_diff_29,
        'mre_p1p2': mre_p1p2,
        'sdr_10px_p1p2': sdr_avg['sdr_10px']
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare original and combined models')
    parser.add_argument('--original_model', type=str, default='checkpoint_best_768.pth')
    parser.add_argument('--combined_model', type=str, default='checkpoint_best_768_combined_31_finetuned.pth')
    parser.add_argument('--annotations_file', type=str, default='annotations_p1_p2.json')
    parser.add_argument('--images_dir', type=str, default='Aariz/train/Cephalograms')
    parser.add_argument('--max_samples', type=int, default=100)
    
    args = parser.parse_args()
    
    try:
        results = test_models_comparison(
            original_model_path=args.original_model,
            combined_model_path=args.combined_model,
            annotations_file=args.annotations_file,
            images_dir=args.images_dir,
            max_samples=args.max_samples
        )
        print("\n[OK] Testing completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()




