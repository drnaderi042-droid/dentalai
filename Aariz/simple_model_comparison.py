"""
Simple model comparison - tests inference speed and basic functionality
"""
import torch
import time
import json
from pathlib import Path
import os

from create_combined_model import SimplifiedCombinedModel
from model_heatmap import HRNetP1P2HeatmapDetector


def test_inference_speed(model, model_name, test_images, num_runs=10):
    """Test inference speed for a model"""
    print(f"\nTesting {model_name} inference speed...")

    model.eval()
    times = []

    with torch.no_grad():
        # Warm up
        for _ in range(3):
            _ = model(test_images[0:1])

        # Actual timing
        for i in range(num_runs):
            start_time = time.time()
            outputs = model(test_images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            inference_time = (end_time - start_time) / len(test_images) * 1000  # ms per image
            times.append(inference_time)
            print(".2f")

    avg_time = sum(times) / len(times)
    std_time = torch.std(torch.tensor(times)).item()

    return {
        'model_name': model_name,
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'min_time_ms': min(times),
        'max_time_ms': max(times),
        'fps': 1000 / avg_time
    }


def test_model_outputs(model, model_name, test_image):
    """Test basic model outputs"""
    print(f"Testing {model_name} outputs...")

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'extract_coordinates'):
            # Heatmap-based model
            heatmaps = model(test_image)
            coords = model.extract_coordinates(heatmaps)
        else:
            # Direct coordinate model
            coords = model(test_image)

    return {
        'model_name': model_name,
        'output_shape': list(coords.shape),
        'output_range': {
            'min': float(coords.min()),
            'max': float(coords.max()),
            'mean': float(coords.mean()),
            'std': float(coords.std())
        },
        'num_landmarks': coords.shape[-1] // 2 if len(coords.shape) > 1 else 1
    }


def main():
    """Main comparison function"""
    print("SIMPLE MODEL COMPARISON")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create test data
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 768, 768).to(device)
    single_test_image = torch.randn(1, 3, 768, 768).to(device)

    results = {
        'comparison_info': {
            'device': str(device),
            'batch_size': batch_size,
            'image_size': '768x768',
            'num_test_runs': 10
        },
        'models': {},
        'performance_comparison': {}
    }

    # Test Combined Model
    print("\n1. Testing Combined Model (31 landmarks)")
    try:
        combined_checkpoint = torch.load('models/combined_31_landmarks.pth', map_location=device)
        combined_model = SimplifiedCombinedModel(
            num_landmarks=31,
            backbone='hrnet_w18',
            output_size=combined_checkpoint.get('heatmap_size', 192)
        )
        combined_model.load_state_dict(combined_checkpoint['model_state_dict'], strict=False)
        combined_model = combined_model.to(device)

        # Test outputs
        output_info = test_model_outputs(combined_model, 'Combined Model', single_test_image)

        # Test speed
        speed_info = test_inference_speed(combined_model, 'Combined Model', test_images)

        results['models']['combined'] = {
            'output_info': output_info,
            'speed_info': speed_info,
            'file_size_mb': Path('models/combined_31_landmarks.pth').stat().st_size / (1024*1024),
            'landmarks': 31
        }

    except Exception as e:
        print(f"Error testing combined model: {e}")
        results['models']['combined'] = {'error': str(e)}

    # Test P1/P2 Model
    print("\n2. Testing P1/P2 Model (2 landmarks)")
    try:
        p1p2_checkpoint = torch.load('models/hrnet_p1p2_heatmap_best.pth', map_location=device)
        p1p2_model = HRNetP1P2HeatmapDetector(
            num_landmarks=2,
            hrnet_variant='hrnet_w18',
            pretrained=False,
            output_size=p1p2_checkpoint.get('heatmap_size', 192)
        )
        p1p2_model.load_state_dict(p1p2_checkpoint['model_state_dict'], strict=False)
        p1p2_model = p1p2_model.to(device)

        # Test outputs
        output_info = test_model_outputs(p1p2_model, 'P1/P2 Model', single_test_image)

        # Test speed
        speed_info = test_inference_speed(p1p2_model, 'P1/P2 Model', test_images)

        results['models']['p1p2'] = {
            'output_info': output_info,
            'speed_info': speed_info,
            'file_size_mb': Path('models/hrnet_p1p2_heatmap_best.pth').stat().st_size / (1024*1024),
            'landmarks': 2
        }

    except Exception as e:
        print(f"Error testing P1/P2 model: {e}")
        results['models']['p1p2'] = {'error': str(e)}

    # Performance comparison
    if 'combined' in results['models'] and 'p1p2' in results['models']:
        combined_speed = results['models']['combined'].get('speed_info', {}).get('avg_inference_time_ms', float('inf'))
        p1p2_speed = results['models']['p1p2'].get('speed_info', {}).get('avg_inference_time_ms', float('inf'))

        combined_size = results['models']['combined'].get('file_size_mb', 0)
        p1p2_size = results['models']['p1p2'].get('file_size_mb', 0)

        results['performance_comparison'] = {
            'speed_comparison': {
                'combined_model_ms': combined_speed,
                'p1p2_model_ms': p1p2_speed,
                'speed_ratio': combined_speed / p1p2_speed if p1p2_speed > 0 else float('inf'),
                'combined_slower_by_ms': combined_speed - p1p2_speed,
                'combined_slower_by_pct': ((combined_speed - p1p2_speed) / p1p2_speed) * 100 if p1p2_speed > 0 else float('inf')
            },
            'size_comparison': {
                'combined_model_mb': combined_size,
                'p1p2_model_mb': p1p2_size,
                'size_ratio': combined_size / p1p2_size if p1p2_size > 0 else float('inf'),
                'combined_larger_by_mb': combined_size - p1p2_size,
                'combined_larger_by_pct': ((combined_size - p1p2_size) / p1p2_size) * 100 if p1p2_size > 0 else float('inf')
            },
            'efficiency_metrics': {
                'combined_ms_per_landmark': combined_speed / 31,
                'p1p2_ms_per_landmark': p1p2_speed / 2,
                'combined_mb_per_landmark': combined_size / 31,
                'p1p2_mb_per_landmark': p1p2_size / 2
            }
        }

    # Save results
    os.makedirs('model_comparison_results', exist_ok=True)
    with open('model_comparison_results/simple_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)

    if 'performance_comparison' in results:
        perf = results['performance_comparison']

        print("Speed Comparison:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".1f")

        print("\nSize Comparison:")
        print(".1f")
        print(".1f")
        print(".2f")
        print(".1f")

        print("\nEfficiency (per landmark):")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

    print(f"\nDetailed results saved to: model_comparison_results/simple_comparison.json")
    print("="*50)


if __name__ == '__main__':
    main()