"""
Inference script for P1/P2 heatmap model
Called from Node.js API endpoint
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import base64
import json
import sys
import argparse
from io import BytesIO
from pathlib import Path
import urllib.request
import cv2

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from model_heatmap import HRNetP1P2HeatmapDetector


def load_image_from_data(image_data, is_base64=True):
    """Load image from base64 string or URL"""
    if is_base64:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    else:
        # Download from URL
        with urllib.request.urlopen(image_data) as response:
            image_bytes = response.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    return image


def preprocess_image(image, image_size=1024):
    """Preprocess image for model"""
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def infer_p1p2(model, image_tensor, image_size=1024, device='cuda'):
    """Run inference"""
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get heatmaps
        heatmaps = model(image_tensor)  # (1, 2, H, W)
        
        # Extract coordinates
        coords = model.extract_coordinates(heatmaps)  # (1, 4) normalized [0, 1]
        
        # Convert to pixel coordinates
        coords_px = coords.cpu().numpy()[0] * image_size
        
        p1 = {
            'x': float(coords_px[0]),
            'y': float(coords_px[1])
        }
        p2 = {
            'x': float(coords_px[2]),
            'y': float(coords_px[3])
        }
        
        # Calculate confidence (max heatmap value)
        heatmap_max = heatmaps.max().item()
        confidence = min(1.0, heatmap_max * 1.2)  # Scale to [0, 1]
        
        return p1, p2, confidence


def main():
    parser = argparse.ArgumentParser(description='P1/P2 Heatmap Inference')
    parser.add_argument('--image', required=True, help='Base64 image or URL')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--image-size', type=int, default=1024, help='Input image size (MUST match training: 1024)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    image_size = checkpoint.get('image_size', args.image_size)
    heatmap_size = checkpoint.get('heatmap_size', 256)  # ✅ Match training configuration (256)
    
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=True,  # Must match training configuration!
        output_size=heatmap_size
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    is_base64 = not args.image.startswith('http')
    image = load_image_from_data(args.image, is_base64=is_base64)
    image_tensor = preprocess_image(image, image_size)
    
    # Run inference
    p1, p2, confidence = infer_p1p2(model, image_tensor, image_size, device)
    
    # Output JSON
    result = {
        'p1': p1,
        'p2': p2,
        'confidence': confidence
    }
    
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == '__main__':
    main()













