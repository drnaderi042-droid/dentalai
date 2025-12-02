"""
Test the exact P1/P2 coordinate transformation logic from unified_ai_api_server.py
"""
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def test_unified_api_coordinate_logic():
    """Test the exact coordinate transformation from unified_ai_api_server.py"""
    print("Testing unified_ai_api_server.py coordinate transformation logic...")
    
    try:
        # Add Aariz directory to path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if os.path.exists(aariz_dir) and aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        
        from model_heatmap import HRNetP1P2HeatmapDetector
        
        # Load model exactly as unified_ai_api_server.py does
        model_path_candidates = [
            os.path.join(base_dir, 'Aariz', 'models', 'hrnet_p1p2_heatmap_best.pth'),
            os.path.join(base_dir, 'aariz', 'models', 'hrnet_p1p2_heatmap_best.pth'),
        ]
        
        model_path = None
        for candidate in model_path_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if not model_path:
            print("Model not found!")
            return
        
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        device = 'cpu'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get parameters from checkpoint (EXACTLY as unified_ai_api_server.py does)
        p1p2_image_size = checkpoint.get('image_size', 1024)
        p1p2_heatmap_size = checkpoint.get('heatmap_size', 256)
        
        print(f"Parameters from checkpoint:")
        print(f"  p1p2_image_size: {p1p2_image_size}")
        print(f"  p1p2_heatmap_size: {p1p2_heatmap_size}")
        
        # Load state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Create model
        model = HRNetP1P2HeatmapDetector(
            num_landmarks=2,
            hrnet_variant='hrnet_w18',
            pretrained=False,
            output_size=p1p2_heatmap_size
        )
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Create a test image
        test_image = Image.new('RGB', (1024, 1024), color='white')
        # Add some landmarks
        img_array = np.array(test_image)
        img_array[800:810, 500:510] = 0  # P1 at bottom
        img_array[200:210, 500:510] = 0  # P2 at top
        test_image = Image.fromarray(img_array)
        
        # Test with different image sizes to simulate real usage
        test_cases = [
            (1024, 1024, "Square image"),
            (800, 600, "Rectangular image"),
            (2048, 1536, "Large image"),
        ]
        
        for orig_width, orig_height, description in test_cases:
            print(f"\n--- Testing {description} ({orig_width}x{orig_height}) ---")
            
            # Resize test image to match case
            resized_image = test_image.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
            
            # EXACTLY replicate the preprocessing from unified_ai_api_server.py
            transform = transforms.Compose([
                transforms.Resize((p1p2_image_size, p1p2_image_size)),  # Resize to 1024x1024
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(resized_image).unsqueeze(0).to(device)
            
            print(f"Image preprocessing:")
            print(f"  Original size: {orig_width}x{orig_height}")
            print(f"  Resized to: {p1p2_image_size}x{p1p2_image_size}")
            print(f"  Tensor shape: {image_tensor.shape}")
            
            # Run inference
            with torch.no_grad():
                heatmaps = model(image_tensor)
                
                # Extract coordinates
                if hasattr(model, 'extract_coordinates'):
                    coords = model.extract_coordinates(heatmaps)
                else:
                    # Manual soft-argmax
                    batch_size, num_landmarks, H, W = heatmaps.shape
                    y_coords = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, H, 1)
                    x_coords = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, 1, W)
                    
                    y_coords = y_coords / (H - 1) if H > 1 else y_coords
                    x_coords = x_coords / (W - 1) if W > 1 else x_coords
                    
                    heatmaps_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
                    x_mean = (heatmaps * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    y_mean = (heatmaps * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    
                    coords = torch.stack([x_mean, y_mean], dim=-1)
                    coords = coords.view(batch_size, num_landmarks * 2)
            
            # EXACTLY replicate the coordinate transformation from unified_ai_api_server.py
            coords_normalized = coords.cpu().numpy()[0]  # [p1_x, p1_y, p2_x, p2_y]
            
            print(f"Model output (normalized [0,1]):")
            print(f"  Raw coords: {coords_normalized}")
            
            # THIS IS THE EXACT TRANSFORMATION FROM unified_ai_api_server.py lines 2984-2987
            x0 = coords_normalized[0] * orig_width
            y0 = coords_normalized[1] * orig_height  
            x1 = coords_normalized[2] * orig_width
            y1 = coords_normalized[3] * orig_height
            
            print(f"Transformed coordinates (original image):")
            print(f"  p1: ({x0:.1f}, {y0:.1f})")
            print(f"  p2: ({x1:.1f}, {y1:.1f})")
            
            # Check if coordinates are reasonable
            max_x = max(x0, x1)
            max_y = max(y0, y1)
            print(f"  Max coordinate: ({max_x:.1f}, {max_y:.1f})")
            
            if max_x > orig_width or max_y > orig_height:
                print(f"  ERROR: Coordinates are OUT OF BOUNDS!")
                print(f"  Image bounds: ({orig_width}, {orig_height})")
                print(f"  Coordinate bounds: ({max_x:.1f}, {max_y:.1f})")
            else:
                print(f"  OK: Coordinates are within image bounds")
                
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_unified_api_coordinate_logic()