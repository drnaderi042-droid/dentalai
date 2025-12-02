"""
Inference Script for Single Image Prediction
برای استفاده در وب اپلیکیشن
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
from utils import heatmap_to_coordinates, load_checkpoint


class LandmarkPredictor:
    """
    کلاس برای پیش‌بینی لندمارک‌ها در یک تصویر
    """
    def __init__(self, checkpoint_path, model_name='resnet', device='cuda'):
        """
        Args:
            checkpoint_path: مسیر فایل checkpoint مدل
            model_name: نام معماری مدل ('resnet', 'unet', 'hourglass', 'hrnet')
            device: 'cuda' یا 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # ابتدا checkpoint را بارگذاری کن تا تعداد لندمارک‌ها را تشخیص دهیم
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # تشخیص تعداد لندمارک‌ها
        num_landmarks = checkpoint.get('num_landmarks', None)
        
        # اگر num_landmarks در checkpoint نیست، از state_dict تشخیص بده
        if num_landmarks is None and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # بررسی final_layers برای تشخیص تعداد لندمارک‌ها
            # HRNet final_layers structure:
            # - final_layers.0: Sequential with 4 layers, last is index 3 (output Conv2d)
            # - final_layers.1-3: Sequential with 5 layers, last is index 4 (output Conv2d)
            # The final output layers are: final_layers.0.3, final_layers.1.4, final_layers.2.4, final_layers.3.4
            # These are 1x1 Conv2d layers that output num_landmarks channels
            
            # First, try to find the final output weight layers (most reliable)
            final_output_patterns = [
                'final_layers.0.3.weight',  # First branch final output
                'final_layers.1.4.weight',   # Second branch final output
                'final_layers.2.4.weight',  # Third branch final output
                'final_layers.3.4.weight'   # Fourth branch final output
            ]
            
            for pattern in final_output_patterns:
                if pattern in state_dict:
                    tensor = state_dict[pattern]
                    if len(tensor.shape) == 4:  # Conv2d weight: [out_channels, in_channels, H, W]
                        out_channels = tensor.shape[0]
                        kernel_size = tensor.shape[2:]  # (H, W)
                        # Verify it's a 1x1 conv (final output layer)
                        if kernel_size == (1, 1):
                            num_landmarks = out_channels
                            print(f"   Detected {num_landmarks} landmarks from checkpoint structure ({pattern})")
                            break
            
            # If weight not found, try bias as fallback (same patterns)
            if num_landmarks is None:
                final_bias_patterns = [
                    'final_layers.0.3.bias',
                    'final_layers.1.4.bias',
                    'final_layers.2.4.bias',
                    'final_layers.3.4.bias'
                ]
                for pattern in final_bias_patterns:
                    if pattern in state_dict:
                        tensor = state_dict[pattern]
                        if len(tensor.shape) == 1:  # Bias: [out_channels]
                            num_landmarks = tensor.shape[0]
                            print(f"   Detected {num_landmarks} landmarks from checkpoint structure ({pattern})")
                            break
            
            if num_landmarks is None:
                # Fallback to default
                num_landmarks = 29
                print(f"   Could not detect landmarks from checkpoint, using default: {num_landmarks}")
        
        # Final fallback
        if num_landmarks is None:
            num_landmarks = 29
        
        self.num_landmarks = num_landmarks
        
        # ایجاد مدل با تعداد لندمارک صحیح
        self.model = get_model(model_name, num_landmarks=num_landmarks)
        self.model = self.model.to(self.device)
        
        # بارگذاری checkpoint (استفاده از checkpoint که قبلاً بارگذاری شده)
        # load_checkpoint دوباره checkpoint را بارگذاری می‌کند، اما ما می‌توانیم مستقیماً state_dict را بارگذاری کنیم
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"✅ Checkpoint loaded successfully with strict=True")
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a size mismatch error
            if 'size mismatch' in error_msg.lower():
                print(f"⚠️  Warning: Size mismatch detected. Filtering incompatible layers...")
                print(f"   Model expects: {self.num_landmarks} landmarks")
                
                # Manually filter state_dict to only include compatible layers
                model_state_dict = self.model.state_dict()
                checkpoint_state_dict = checkpoint['model_state_dict']
                filtered_state_dict = {}
                skipped_keys = []
                
                for key in checkpoint_state_dict.keys():
                    if key in model_state_dict:
                        checkpoint_shape = checkpoint_state_dict[key].shape
                        model_shape = model_state_dict[key].shape
                        if checkpoint_shape == model_shape:
                            filtered_state_dict[key] = checkpoint_state_dict[key]
                        else:
                            skipped_keys.append(f"{key}: checkpoint {checkpoint_shape} vs model {model_shape}")
                    else:
                        # Key not in model, skip it
                        skipped_keys.append(f"{key}: not in model")
                
                if skipped_keys:
                    print(f"   ⚠️  Skipped {len(skipped_keys)} incompatible layers")
                    if len(skipped_keys) <= 10:  # Only print details if not too many
                        for skip_info in skipped_keys:
                            print(f"      - {skip_info}")
                
                # Load the filtered state_dict
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
                    if missing_keys:
                        print(f"   ⚠️  Missing keys (not loaded): {len(missing_keys)} keys")
                        if len(missing_keys) <= 5:
                            for mk in missing_keys:
                                print(f"      - {mk}")
                    if unexpected_keys:
                        print(f"   ⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
                    print(f"✅ Checkpoint loaded successfully (filtered {len(skipped_keys)} incompatible layers)")
                except RuntimeError as e2:
                    print(f"❌ Error: Failed to load filtered checkpoint: {e2}")
                    raise RuntimeError(f"Failed to load checkpoint. Model initialized with {self.num_landmarks} landmarks, but checkpoint appears to have a different number. Original error: {error_msg}")
            else:
                # Other RuntimeError, try strict=False anyway
                print(f"⚠️  Warning: Strict loading failed: {e}")
                print(f"   Attempting to load with strict=False...")
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if missing_keys:
                        print(f"   ⚠️  Missing keys (not loaded): {len(missing_keys)} keys")
                    if unexpected_keys:
                        print(f"   ⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
                    print(f"✅ Checkpoint loaded with strict=False")
                except RuntimeError as e2:
                    print(f"❌ Error: Even strict=False loading failed: {e2}")
                    raise
        
        self.model.eval()
        
        # نام لندمارک‌ها - 29 لندمارک آناتومیک
        base_landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        
        # اگر 31 لندمارک داریم، P1 و P2 را اضافه کن
        if num_landmarks == 31:
            self.landmark_symbols = base_landmark_symbols + ["p1", "p2"]
        elif num_landmarks == 29:
            self.landmark_symbols = base_landmark_symbols
        else:
            # برای سایر تعدادها، فقط base symbols را استفاده کن
            self.landmark_symbols = base_landmark_symbols[:num_landmarks] if num_landmarks <= len(base_landmark_symbols) else base_landmark_symbols + [f"landmark_{i}" for i in range(len(base_landmark_symbols), num_landmarks)]
        
        print(f"Model loaded successfully on {self.device} with {num_landmarks} landmarks")
    
    def preprocess_image(self, image, target_size=(256, 256)):
        """
        پیش‌پردازش تصویر ورودی
        
        Args:
            image: PIL Image یا numpy array
            target_size: اندازه خروجی (height, width)
        
        Returns:
            tensor: تصویر preprocess شده
            original_size: اندازه اصلی تصویر
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # ذخیره اندازه اصلی
        original_size = image.size  # (width, height)
        
        # تبدیل به RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # تبدیل به tensor و normalize
        # CRITICAL: Match training normalization (mean=0.5, std=0.5)
        # This converts [0, 1] to [-1, 1] range, same as training
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] and CHW format
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training
        ])
        
        image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_size
    
    def predict(self, image, target_size=(256, 256), return_heatmaps=False):
        """
        پیش‌بینی لندمارک‌ها برای یک تصویر
        
        Args:
            image: PIL Image یا numpy array
            target_size: اندازه تصویر برای مدل (height, width)
            return_heatmaps: اگر True باشد، heatmap ها را هم برمی‌گرداند
        
        Returns:
            landmarks: دیکشنری با نام لندمارک و مختصات (x, y)
            landmarks_normalized: مختصات normalize شده (0-1)
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image, target_size)
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            heatmaps = self.model(image_tensor)
            
            # اگر heatmap ها کوچک‌تر از تصویر بودند، resize کن
            if heatmaps.shape[2:] != (target_size[0], target_size[1]):
                heatmaps = F.interpolate(
                    heatmaps,
                    size=(target_size[0], target_size[1]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # تبدیل به numpy
            heatmaps_np = torch.sigmoid(heatmaps).cpu().numpy()[0]  # Remove batch dimension
        
        # تبدیل heatmap به مختصات
        h, w = heatmaps_np.shape[1], heatmaps_np.shape[2]
        coordinates = heatmap_to_coordinates(heatmaps_np, h, w)
        
        # Scale به اندازه اصلی تصویر
        scale_x = original_size[0] / w  # original_width / model_width
        scale_y = original_size[1] / h  # original_height / model_height
        
        coordinates_scaled = coordinates.copy()
        coordinates_scaled[:, 0] *= scale_x
        coordinates_scaled[:, 1] *= scale_y
        
        # Normalize coordinates (0-1)
        coordinates_normalized = coordinates.copy()
        coordinates_normalized[:, 0] /= w
        coordinates_normalized[:, 1] /= h
        
        # ساخت دیکشنری با نام لندمارک‌ها
        landmarks_dict = {}
        landmarks_normalized_dict = {}
        
        for i, symbol in enumerate(self.landmark_symbols):
            if coordinates_scaled[i, 0] >= 0 and coordinates_scaled[i, 1] >= 0:
                landmarks_dict[symbol] = {
                    'x': float(coordinates_scaled[i, 0]),
                    'y': float(coordinates_scaled[i, 1])
                }
                landmarks_normalized_dict[symbol] = {
                    'x': float(coordinates_normalized[i, 0]),
                    'y': float(coordinates_normalized[i, 1])
                }
            else:
                landmarks_dict[symbol] = None
                landmarks_normalized_dict[symbol] = None
        
        # 🔧 FIX: Map AARIZ landmark names to CLDETECTION names for compatibility
        # LIT → L1, LIA → L1A, UIA → U1A, UIT → U1
        mapped_landmarks = {}
        mapped_landmarks_normalized = {}
        
        # Mapping rules: AARIZ names → CLDETECTION names
        landmark_mapping = {
            'LIT': 'L1',   # Lower Incisor Tip → L1
            'LIA': 'L1A',  # Lower Incisor Apex → L1A
            'UIA': 'U1A',  # Upper Incisor Apex → U1A
            'UIT': 'U1',   # Upper Incisor Tip → U1
        }
        
        # Map landmarks according to the mapping rules
        for symbol, landmark_data in landmarks_dict.items():
            if symbol in landmark_mapping:
                mapped_name = landmark_mapping[symbol]
                mapped_landmarks[mapped_name] = landmark_data
                mapped_landmarks_normalized[mapped_name] = landmarks_normalized_dict[symbol]
            else:
                # Keep other landmarks as-is
                mapped_landmarks[symbol] = landmark_data
                mapped_landmarks_normalized[symbol] = landmarks_normalized_dict[symbol]
        
        result = {
            'landmarks': mapped_landmarks,
            'landmarks_normalized': mapped_landmarks_normalized,
            'image_size': original_size
        }
        
        if return_heatmaps:
            result['heatmaps'] = heatmaps_np
        
        return result
    
    def predict_batch(self, images, target_size=(256, 256)):
        """
        پیش‌بینی برای چندین تصویر به صورت batch
        
        Args:
            images: لیستی از PIL Images یا numpy arrays
            target_size: اندازه تصویر برای مدل
        
        Returns:
            list: لیست نتایج برای هر تصویر
        """
        results = []
        for image in images:
            result = self.predict(image, target_size)
            results.append(result)
        return results


def main():
    """
    مثال استفاده از LandmarkPredictor
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict landmarks on a single image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='resnet', help='Model architecture')
    parser.add_argument('--output', type=str, default='prediction.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # ایجاد predictor
    predictor = LandmarkPredictor(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=args.device
    )
    
    # بارگذاری تصویر
    image = Image.open(args.image)
    
    # پیش‌بینی
    print("Predicting landmarks...")
    result = predictor.predict(image)
    
    # ذخیره نتایج
    import json
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Found {sum(1 for v in result['landmarks'].values() if v is not None)} valid landmarks")


if __name__ == "__main__":
    main()


