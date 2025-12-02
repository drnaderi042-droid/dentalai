"""
پیاده‌سازی Test-Time Augmentation برای بهبود دقت
"""

import torch
import numpy as np
from PIL import Image
from inference import LandmarkPredictor

class TTAPredictor:
    """Predictor with Test-Time Augmentation"""
    
    def __init__(self, checkpoint_path, model_name='hrnet', device='cuda'):
        self.predictor = LandmarkPredictor(checkpoint_path, model_name, device)
        self.device = self.predictor.device
    
    def predict_with_tta(self, image, target_size=(256, 256), use_flip=True, use_rotation=False):
        """
        پیش‌بینی با Test-Time Augmentation
        
        Args:
            image: PIL Image یا numpy array
            target_size: اندازه تصویر برای مدل
            use_flip: استفاده از horizontal flip
            use_rotation: استفاده از rotation (90, 180, 270)
        
        Returns:
            landmarks: دیکشنری با نام لندمارک و مختصات (x, y)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size  # (width, height)
        w, h = original_size
        
        predictions = []
        weights = []
        
        # 1. Original image
        pred_orig = self.predictor.predict(image, target_size)
        predictions.append(pred_orig)
        weights.append(1.0)
        
        # 2. Horizontal flip
        if use_flip:
            img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            pred_flip = self.predictor.predict(img_flip, target_size)
            
            # Flip coordinates back
            pred_flip_flipped = {}
            for name, coords in pred_flip['landmarks'].items():
                # Scale coordinates back to original size
                scale_x = w / target_size[1]
                scale_y = h / target_size[0]
                x_scaled = coords['x'] * scale_x
                y_scaled = coords['y'] * scale_y
                
                # Flip x coordinate
                pred_flip_flipped[name] = {
                    'x': w - x_scaled,
                    'y': y_scaled
                }
            
            predictions.append(pred_flip_flipped)
            weights.append(1.0)
        
        # 3. Rotations (optional)
        if use_rotation:
            for angle in [90, 180, 270]:
                img_rot = image.rotate(-angle, expand=False)  # Negative for counter-clockwise
                pred_rot = self.predictor.predict(img_rot, target_size)
                
                # Rotate coordinates back
                pred_rot_rotated = {}
                for name, coords in pred_rot['landmarks'].items():
                    scale_x = w / target_size[1]
                    scale_y = h / target_size[0]
                    x_scaled = coords['x'] * scale_x
                    y_scaled = coords['y'] * scale_y
                    
                    # Rotate back (counter-clockwise)
                    if angle == 90:
                        x_new = y_scaled
                        y_new = w - x_scaled
                    elif angle == 180:
                        x_new = w - x_scaled
                        y_new = h - y_scaled
                    elif angle == 270:
                        x_new = h - y_scaled
                        y_new = x_scaled
                    else:
                        x_new, y_new = x_scaled, y_scaled
                    
                    pred_rot_rotated[name] = {'x': x_new, 'y': y_new}
                
                predictions.append(pred_rot_rotated)
                weights.append(0.5)  # Lower weight for rotations
        
        # Weighted averaging
        final_landmarks = {}
        for name in predictions[0].keys():
            xs = []
            ys = []
            for pred, weight in zip(predictions, weights):
                if name in pred:
                    xs.extend([pred[name]['x']] * int(weight * 10))  # Weight as frequency
                    ys.extend([pred[name]['y']] * int(weight * 10))
            
            if xs:
                final_landmarks[name] = {
                    'x': np.mean(xs),
                    'y': np.mean(ys),
                    'confidence': 1.0  # Average confidence
                }
        
        return {
            'landmarks': final_landmarks,
            'image_size': original_size,
            'num_augmentations': len(predictions)
        }


def test_tta():
    """تست TTA"""
    import os
    
    # Paths
    checkpoint_path = "checkpoints/checkpoint_best.pth"
    test_image_path = "Aariz/train/Cephalograms/cks2ip8fq29yq0yufc4scftj8.png"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not os.path.exists(test_image_path):
        print(f"❌ Image not found: {test_image_path}")
        return
    
    # Load image
    img = Image.open(test_image_path).convert('RGB')
    
    # Without TTA
    print("="*80)
    print("🔬 تست بدون TTA")
    print("="*80)
    predictor = LandmarkPredictor(checkpoint_path, model_name='hrnet')
    result_no_tta = predictor.predict(img, target_size=(256, 256))
    print(f"✅ پیش‌بینی بدون TTA انجام شد")
    
    # With TTA
    print("\n" + "="*80)
    print("🔬 تست با TTA")
    print("="*80)
    tta_predictor = TTAPredictor(checkpoint_path, model_name='hrnet')
    result_tta = tta_predictor.predict_with_tta(img, target_size=(256, 256))
    print(f"✅ پیش‌بینی با TTA انجام شد")
    print(f"   تعداد augmentations: {result_tta['num_augmentations']}")
    
    # Compare (sample)
    print("\n" + "="*80)
    print("📊 مقایسه (نمونه - 5 لندمارک اول)")
    print("="*80)
    landmarks_no_tta = result_no_tta['landmarks']
    landmarks_tta = result_tta['landmarks']
    
    for name in list(landmarks_no_tta.keys())[:5]:
        if name in landmarks_tta:
            no_tta = landmarks_no_tta[name]
            tta = landmarks_tta[name]
            diff_x = abs(no_tta['x'] - tta['x'])
            diff_y = abs(no_tta['y'] - tta['y'])
            diff_total = np.sqrt(diff_x**2 + diff_y**2)
            
            print(f"{name}:")
            print(f"   بدون TTA: ({no_tta['x']:.2f}, {no_tta['y']:.2f})")
            print(f"   با TTA:    ({tta['x']:.2f}, {tta['y']:.2f})")
            print(f"   تفاوت:     {diff_total:.2f} px")


if __name__ == '__main__':
    test_tta()

