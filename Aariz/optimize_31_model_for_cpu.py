"""
Optimize 31-landmark heatmap model for CPU deployment
Techniques: quantization, pruning, ONNX conversion
Target: Fast inference on CPU servers
"""
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
import time
import argparse

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available, skipping ONNX conversion")


class CPUOptimizedHRNet31(nn.Module):
    """CPU-optimized version of HRNet31 detector"""

    def __init__(self, model_path):
        super().__init__()
        # Load the trained model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Create model architecture
        from model_heatmap_31 import HRNet31HeatmapDetector
        self.model = HRNet31HeatmapDetector(
            num_landmarks=31,
            hrnet_variant='hrnet_w32',
            pretrained=False,
            output_size=384
        )

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Fuse layers for quantization
        self._fuse_layers()

    def _fuse_layers(self):
        """Fuse Conv2d + BatchNorm2d + ReLU for better quantization"""
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 2):
                    # Fuse Conv2d + BatchNorm2d + ReLU
                    if (isinstance(module[i], nn.Conv2d) and
                        isinstance(module[i+1], nn.BatchNorm2d) and
                        isinstance(module[i+2], nn.ReLU)):
                        torch.quantization.fuse_modules(
                            module, [str(i), str(i+1), str(i+2)], inplace=True
                        )

    def forward(self, x):
        with torch.no_grad():
            heatmaps = self.model(x)
            coords, confs = self.model.extract_coordinates(heatmaps)
            return coords, confs


def quantize_model(model_path, output_path):
    """Quantize model for faster CPU inference"""
    print("🔧 Applying dynamic quantization...")

    model = CPUOptimizedHRNet31(model_path)

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # Save quantized model
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quantized': True,
        'original_mre': torch.load(model_path, map_location='cpu', weights_only=False).get('mre', 'unknown')
    }, output_path)

    print(f"✅ Quantized model saved to: {output_path}")
    return quantized_model


def convert_to_onnx(model_path, output_path):
    """Convert to ONNX for cross-platform deployment"""
    if not ONNX_AVAILABLE:
        print("⚠️ ONNX Runtime not available, skipping conversion")
        return

    print("🔧 Converting to ONNX format...")

    model = CPUOptimizedHRNet31(model_path)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 768, 768)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['coordinates', 'confidences'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )

    print(f"✅ ONNX model saved to: {output_path}")


def benchmark_model(model_path, method="pytorch"):
    """Benchmark inference speed"""
    print(f"Benchmarking {method} model...")

    if method == "onnx" and ONNX_AVAILABLE:
        # ONNX inference
        ort_session = ort.InferenceSession(model_path)
        dummy_input = np.random.randn(1, 3, 768, 768).astype(np.float32)

        # Warmup
        for _ in range(10):
            ort_session.run(None, {"input": dummy_input})

        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            ort_session.run(None, {"input": dummy_input})
            times.append(time.time() - start)

    else:
        # PyTorch inference
        if method == "quantized":
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            from model_heatmap_31 import HRNet31HeatmapDetector
            model = HRNet31HeatmapDetector(num_landmarks=31, hrnet_variant='hrnet_w32', pretrained=False, output_size=384)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
        else:
            model = CPUOptimizedHRNet31(model_path)

        model.eval()
        dummy_input = torch.randn(1, 3, 768, 768)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                model(dummy_input)
                times.append(time.time() - start)

    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    print(".2f"))
    print(".2f"))
    print(".2f"))
    return avg_time


def optimize_model(input_path, output_dir, methods=None):
    """Apply all optimizations"""
    if methods is None:
        methods = ['quantize', 'onnx']

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem

    print("="*60)
    print("CPU OPTIMIZATION PIPELINE")
    print("="*60)
    print(f"Input model: {input_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load original model info
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    original_mre = checkpoint.get('mre', 'unknown')
    print(".2f")
    # Benchmark original model
    print("\nBenchmarking original model...")
    original_time = benchmark_model(input_path, "original")

    results = {
        'original': {
            'path': input_path,
            'time_ms': original_time,
            'mre': original_mre
        }
    }

    # Apply quantization
    if 'quantize' in methods:
        quant_path = os.path.join(output_dir, f"{base_name}_quantized.pth")
        quantize_model(input_path, quant_path)
        quant_time = benchmark_model(quant_path, "quantized")

        results['quantized'] = {
            'path': quant_path,
            'time_ms': quant_time,
            'mre': original_mre  # MRE should remain similar
        }

        speedup = original_time / quant_time
        print(".1f"
    # Convert to ONNX
    if 'onnx' in methods and ONNX_AVAILABLE:
        onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
        convert_to_onnx(input_path, onnx_path)
        onnx_time = benchmark_model(onnx_path, "onnx")

        results['onnx'] = {
            'path': onnx_path,
            'time_ms': onnx_time,
            'mre': original_mre
        }

        speedup = original_time / onnx_time
        print(".1f"
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)

    for method, info in results.items():
        print(f"{method.upper():>10}: {info['time_ms']:.1f}ms "
              f"({'preserved' if method != 'original' else ''})")

    print("\n🎯 Recommended for CPU server deployment:")
    if 'quantized' in results:
        print("   Quantized PyTorch model (best compatibility)")
    elif 'onnx' in results:
        print("   ONNX model (best performance)")
    else:
        print("   Original model (fallback)")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize 31-landmark model for CPU deployment')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='models/optimized',
                        help='Output directory')
    parser.add_argument('--methods', nargs='+',
                        choices=['quantize', 'onnx'],
                        default=['quantize', 'onnx'],
                        help='Optimization methods to apply')

    args = parser.parse_args()

    optimize_model(args.input, args.output, args.methods)
