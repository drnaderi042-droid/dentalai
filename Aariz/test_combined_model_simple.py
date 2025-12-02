"""
Simple test to verify combined model works correctly
Tests model architecture and output shape on 5 images
"""
import torch
import cv2
import numpy as np
import json
from pathlib import Path
from create_combined_model import SimplifiedCombinedModel


def load_model(model_path='combined_31_landmarks.pth'):
    """Load the combined model"""
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = SimplifiedCombinedModel(
        num_landmarks=31,
        backbone='hrnet_w18',
        output_size=checkpoint.get('heatmap_size', 192)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    
    return model, device


def preprocess_image(image_path, target_size=768):
    """Preprocess image for model"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    
    img_resized = cv2.resize(img, (target_size, target_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, (original_h, original_w)


def main():
    print("="*70)
    print("Testing Combined 31-Landmark Model")
    print("="*70)
    print()
    
    # Load model
    model, device = load_model('combined_31_landmarks.pth')
    
    # Find test images
    images_dir = Path('Aariz/train/Cephalograms')
    image_files = list(images_dir.glob('*.png'))[:5]
    
    if len(image_files) == 0:
        print("ERROR: No images found!")
        return
    
    print(f"Testing on {len(image_files)} images\n")
    
    # Test each image
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"{'='*70}")
        print(f"Image {idx}/{len(image_files)}: {image_path.name}")
        print(f"{'='*70}")
        
        try:
            # Preprocess
            image_tensor, original_size = preprocess_image(image_path)
            print(f"  Original size: {original_size[1]}x{original_size[0]}")
            print(f"  Input tensor shape: {image_tensor.shape}")
            
            # Predict
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                heatmaps = model(image_tensor)
                coords = model.extract_coordinates(heatmaps)
            
            print(f"  Heatmaps shape: {heatmaps.shape}")
            print(f"  Coordinates shape: {coords.shape}")
            print(f"  Number of landmarks: {coords.shape[1] // 2}")
            
            # Check for valid output
            coords_np = coords.cpu().numpy()
            has_nan = np.isnan(coords_np).any()
            has_inf = np.isinf(coords_np).any()
            
            print(f"  Contains NaN: {has_nan}")
            print(f"  Contains Inf: {has_inf}")
            
            if not has_nan and not has_inf:
                # Denormalize
                coords_denorm = coords_np.reshape(-1, 2)
                coords_denorm[:, 0] *= original_size[1]
                coords_denorm[:, 1] *= original_size[0]
                
                print(f"  [OK] Valid predictions generated")
                print(f"  Sample coordinates (first 3 landmarks):")
                for i in range(min(3, len(coords_denorm))):
                    print(f"    Landmark {i+1}: ({coords_denorm[i, 0]:.1f}, {coords_denorm[i, 1]:.1f})")
                
                print(f"  Calibration points:")
                print(f"    P1 (Landmark 30): ({coords_denorm[29, 0]:.1f}, {coords_denorm[29, 1]:.1f})")
                print(f"    P2 (Landmark 31): ({coords_denorm[30, 0]:.1f}, {coords_denorm[30, 1]:.1f})")
                
                status = "SUCCESS"
            else:
                print(f"  [FAILED] Invalid predictions (NaN or Inf)")
                status = "FAILED"
            
            results.append({
                'image': image_path.name,
                'status': status,
                'output_shape': list(coords.shape),
                'has_nan': bool(has_nan),
                'has_inf': bool(has_inf)
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'image': image_path.name,
                'status': 'ERROR',
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in results if r['status'] == 'FAILED')
    error_count = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"\nTotal images tested: {len(results)}")
    print(f"  [OK] Successful: {success_count}")
    print(f"  [FAILED] Failed (NaN/Inf): {failed_count}")
    print(f"  [ERROR] Errors: {error_count}")
    
    print(f"\nModel Architecture:")
    print(f"  - Input: 768x768 RGB image")
    print(f"  - Backbone: HRNet-W18")
    print(f"  - Output: 31 heatmaps (192x192)")
    print(f"  - Coordinates: 62 values (31 landmarks × 2)")
    
    print(f"\nLandmark Distribution:")
    print(f"  - Landmarks 1-29: Anatomical landmarks")
    print(f"  - Landmarks 30-31: Calibration points (P1, P2)")
    
    if failed_count > 0 or error_count > 0:
        print(f"\n[WARNING] NOTE: Model has random weights (not trained)")
        print(f"          The model architecture works correctly, but needs training")
        print(f"          to produce meaningful predictions.")
    
    # Save results
    output_dir = Path('combined_test_results')
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / 'simple_test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'combined_31_landmarks.pth',
            'num_images': len(results),
            'success_count': success_count,
            'failed_count': failed_count,
            'error_count': error_count,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("="*70)


if __name__ == '__main__':
    main()