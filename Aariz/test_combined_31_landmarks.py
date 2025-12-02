"""
Test script for the combined 31-landmark model
Demonstrates how to load and use the combined model for inference
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
import numpy as np
from create_combined_model import SimplifiedCombinedModel


def load_combined_model(model_path='combined_31_landmarks.pth', device='cuda'):
    """Load the combined 31-landmark model"""
    print(f"Loading combined model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = SimplifiedCombinedModel(
        num_landmarks=31,
        backbone='hrnet_w18',
        output_size=checkpoint.get('heatmap_size', 192)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print("Model loaded on GPU")
    else:
        device = 'cpu'
        print("Model loaded on CPU")
    
    return model, device


def preprocess_image(image_path, target_size=768):
    """Preprocess image for model input"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_h, original_w = img.shape[:2]
    
    # Resize to target size
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, (original_h, original_w)


def predict_landmarks(model, image_tensor, device='cuda'):
    """Predict landmarks from image tensor"""
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass - get heatmaps
        heatmaps = model(image_tensor)
        
        # Extract coordinates from heatmaps
        coords = model.extract_coordinates(heatmaps)
        
        return coords.cpu().numpy(), heatmaps.cpu().numpy()


def denormalize_coordinates(coords, original_size):
    """Convert normalized coordinates [0,1] to pixel coordinates"""
    original_h, original_w = original_size
    
    # coords shape: (1, 62) -> (31, 2)
    coords = coords.reshape(-1, 2)
    
    # Denormalize
    coords[:, 0] *= original_w  # x coordinates
    coords[:, 1] *= original_h  # y coordinates
    
    return coords


def visualize_landmarks(image_path, landmarks, output_path=None):
    """Visualize landmarks on image"""
    # Read original image
    img = cv2.imread(str(image_path))
    
    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        # Different colors for different landmark groups
        if i < 29:
            # Anatomical landmarks (blue)
            color = (255, 0, 0)
            radius = 3
        else:
            # Calibration points P1/P2 (red, larger)
            color = (0, 0, 255)
            radius = 5
        
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
        
        # Add landmark number
        cv2.putText(img, str(i+1), (int(x)+5, int(y)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"Visualization saved to: {output_path}")
    else:
        cv2.imshow('31 Landmarks', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img


def main():
    print("="*60)
    print("Combined 31-Landmark Model Test")
    print("="*60)
    
    # Check if model exists
    model_path = 'combined_31_landmarks.pth'
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Please run create_combined_model.py first")
        return
    
    # Load model
    model, device = load_combined_model(model_path)
    
    # Test with a sample image (you can change this path)
    test_image_path = 'Aariz/train/Cephalograms/001.png'
    
    if not Path(test_image_path).exists():
        print(f"\nWARNING: Test image not found: {test_image_path}")
        print("Creating dummy test to verify model works...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 768, 768).to(device)
        
        with torch.no_grad():
            heatmaps = model(dummy_input)
            coords = model.extract_coordinates(heatmaps)
        
        print(f"\nDummy test successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Heatmaps shape: {heatmaps.shape}")
        print(f"Coordinates shape: {coords.shape}")
        print(f"Number of landmarks: {coords.shape[1] // 2}")
        
        print("\n" + "="*60)
        print("Model is ready for inference!")
        print("="*60)
        print("\nTo test with real images:")
        print("1. Update test_image_path in this script")
        print("2. Run: python test_combined_31_landmarks.py")
        
        return
    
    # Preprocess image
    print(f"\nProcessing image: {test_image_path}")
    image_tensor, original_size = preprocess_image(test_image_path)
    
    # Predict landmarks
    print("Predicting landmarks...")
    coords, heatmaps = predict_landmarks(model, image_tensor, device)
    
    # Denormalize coordinates
    landmarks = denormalize_coordinates(coords, original_size)
    
    print(f"\nPrediction complete!")
    print(f"Detected {len(landmarks)} landmarks")
    print(f"  - Anatomical landmarks (1-29): {landmarks[:29].shape[0]}")
    print(f"  - Calibration points (30-31): {landmarks[29:].shape[0]}")
    
    # Print first few landmarks
    print("\nFirst 5 landmarks (x, y):")
    for i in range(min(5, len(landmarks))):
        print(f"  Landmark {i+1}: ({landmarks[i, 0]:.1f}, {landmarks[i, 1]:.1f})")
    
    print("\nCalibration points (P1, P2):")
    print(f"  P1 (Landmark 30): ({landmarks[29, 0]:.1f}, {landmarks[29, 1]:.1f})")
    print(f"  P2 (Landmark 31): ({landmarks[30, 0]:.1f}, {landmarks[30, 1]:.1f})")
    
    # Visualize
    output_path = 'combined_test_results/visualization_31_landmarks.png'
    visualize_landmarks(test_image_path, landmarks, output_path)
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()