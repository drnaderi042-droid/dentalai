"""
Simple CPU optimization for 31-landmark model
"""
import torch
import torch.nn as nn
import os

def optimize_for_cpu(input_path, output_path):
    """Simple quantization for CPU deployment"""
    print("Loading model...")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # Create model architecture
    from model_heatmap_31 import HRNet31HeatmapDetector
    model = HRNet31HeatmapDetector(
        num_landmarks=31,
        hrnet_variant='hrnet_w32',
        pretrained=False,
        output_size=384
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Applying quantization...")
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    print("Saving quantized model...")
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quantized': True,
        'original_mre': checkpoint.get('mre', 'unknown')
    }, output_path)

    print(f"Quantized model saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    input_path = 'models/hrnet_31_heatmap_best.pth'
    output_path = 'models/hrnet_31_heatmap_quantized.pth'

    if os.path.exists(input_path):
        optimize_for_cpu(input_path, output_path)
        print("CPU optimization completed!")
    else:
        print(f"Model not found: {input_path}")


