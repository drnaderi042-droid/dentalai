"""
Compare accuracy of Combined Model vs Individual Models
Tests all three models (main, P1/P2, combined) on the same dataset and compares results
"""
import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from pathlib import Path
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from create_combined_model import SimplifiedCombinedModel
from model_heatmap import HRNetP1P2HeatmapDetector
from train_p1_p2_heatmap import P1P2HeatmapDataset


class ModelComparator:
    """Compare accuracy of different models on the same dataset"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_models(self):
        """Load all three models"""
        print("Loading models for comparison...")

        # 1. Combined Model (31 landmarks)
        print("1. Loading Combined Model (31 landmarks)...")
        combined_checkpoint = torch.load('models/combined_31_landmarks.pth', map_location=self.device)
        combined_model = SimplifiedCombinedModel(
            num_landmarks=31,
            backbone='hrnet_w18',
            output_size=combined_checkpoint.get('heatmap_size', 192)
        )
        combined_model.load_state_dict(combined_checkpoint['model_state_dict'], strict=False)
        combined_model = combined_model.to(self.device).eval()
        self.models['combined'] = combined_model

        # 2. Main Model (29 landmarks) - Placeholder since we don't have the exact architecture
        print("2. Main Model (29 landmarks) - Using placeholder (coordinates not available)")
        self.models['main'] = None  # We can't load the main model without its architecture

        # 3. P1/P2 Model (2 landmarks)
        print("3. Loading P1/P2 Model (2 landmarks)...")
        p1p2_checkpoint = torch.load('models/hrnet_p1p2_heatmap_best.pth', map_location=self.device)
        p1p2_model = HRNetP1P2HeatmapDetector(
            num_landmarks=2,
            hrnet_variant='hrnet_w18',
            pretrained=False,
            output_size=p1p2_checkpoint.get('heatmap_size', 192)
        )
        p1p2_model.load_state_dict(p1p2_checkpoint['model_state_dict'], strict=False)
        p1p2_model = p1p2_model.to(self.device).eval()
        self.models['p1p2'] = p1p2_model

        print(f"Models loaded successfully on {self.device}")

    def create_comparison_dataset(self, annotations_file, images_dir, num_samples=20):
        """Create dataset for comparison testing"""
        print(f"Creating comparison dataset with {num_samples} samples...")

        # Load P1/P2 dataset (which has both anatomical and calibration landmarks)
        full_dataset = P1P2HeatmapDataset(
            annotations_file, images_dir,
            image_size=768,
            heatmap_size=192,
            augment=False
        )

        # Use validation set or subset
        if len(full_dataset) > 50:
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            _, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            dataset = val_dataset
        else:
            dataset = torch.utils.data.Subset(full_dataset, range(min(num_samples, len(full_dataset))))

        print(f"Using {len(dataset)} samples for comparison")
        return dataset

    def test_combined_model(self, dataset, output_dir):
        """Test combined model accuracy"""
        print("\nTesting Combined Model (31 landmarks)...")

        model = self.models['combined']
        results = []

        for idx, (images, gt_heatmaps, gt_coords) in enumerate(tqdm(dataset, desc="Combined Model")):
            images = images.unsqueeze(0).to(self.device)  # Add batch dimension

            with torch.no_grad():
                pred_heatmaps = model(images)
                pred_coords = model.extract_coordinates(pred_heatmaps)

            # Convert to numpy
            pred_coords_np = pred_coords.cpu().numpy().flatten()
            gt_coords_np = gt_coords.numpy()

            # Calculate errors for all 31 landmarks
            errors = []
            for i in range(31):  # 31 landmarks
                pred_x, pred_y = pred_coords_np[i*2], pred_coords_np[i*2+1]
                gt_x, gt_y = gt_coords_np[i*2], gt_coords_np[i*2+1]

                error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                errors.append(error)

            results.append({
                'sample_idx': idx,
                'pred_coords': pred_coords_np.tolist(),
                'gt_coords': gt_coords_np.tolist(),
                'errors': errors,
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'max_error': np.max(errors)
            })

        self.results['combined'] = results
        print(".2f")

        # Save detailed results
        with open(f"{output_dir}/combined_model_detailed.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def test_p1p2_model(self, dataset, output_dir):
        """Test P1/P2 model accuracy (only for P1/P2 landmarks)"""
        print("\nTesting P1/P2 Model (2 landmarks)...")

        model = self.models['p1p2']
        results = []

        for idx, (images, gt_heatmaps, gt_coords) in enumerate(tqdm(dataset, desc="P1/P2 Model")):
            images = images.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_heatmaps = model(images)
                pred_coords = model.extract_coordinates(pred_heatmaps)

            # Only compare P1/P2 landmarks (last 4 values in gt_coords)
            pred_coords_np = pred_coords.cpu().numpy().flatten()
            gt_p1p2_coords = gt_coords[-4:].numpy()  # Last 4 values are P1/P2

            # Calculate errors for P1/P2 only
            p1_error = np.sqrt((pred_coords_np[0] - gt_p1p2_coords[0])**2 +
                              (pred_coords_np[1] - gt_p1p2_coords[1])**2)
            p2_error = np.sqrt((pred_coords_np[2] - gt_p1p2_coords[2])**2 +
                              (pred_coords_np[3] - gt_p1p2_coords[3])**2)

            results.append({
                'sample_idx': idx,
                'pred_p1': pred_coords_np[:2].tolist(),
                'pred_p2': pred_coords_np[2:].tolist(),
                'gt_p1': gt_p1p2_coords[:2].tolist(),
                'gt_p2': gt_p1p2_coords[2:].tolist(),
                'p1_error': p1_error,
                'p2_error': p2_error,
                'mean_error': (p1_error + p2_error) / 2
            })

        self.results['p1p2'] = results
        p1p2_errors = [r['mean_error'] for r in results]
        print(".2f")

        # Save detailed results
        with open(f"{output_dir}/p1p2_model_detailed.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def generate_comparison_report(self, output_dir):
        """Generate comprehensive comparison report"""
        print("\nGenerating Comparison Report...")

        os.makedirs(output_dir, exist_ok=True)

        # Collect all results
        combined_results = self.results.get('combined', [])
        p1p2_results = self.results.get('p1p2', [])

        if not combined_results or not p1p2_results:
            print(" Not enough results for comparison")
            return

        # Extract error metrics
        combined_errors = [r['mean_error'] for r in combined_results]
        p1p2_errors = [r['mean_error'] for r in p1p2_results]

        # Calculate statistics
        stats = {
            'combined_model': {
                'mean_error': float(np.mean(combined_errors)),
                'median_error': float(np.median(combined_errors)),
                'std_error': float(np.std(combined_errors)),
                'min_error': float(np.min(combined_errors)),
                'max_error': float(np.max(combined_errors)),
                'samples': len(combined_errors)
            },
            'p1p2_model': {
                'mean_error': float(np.mean(p1p2_errors)),
                'median_error': float(np.median(p1p2_errors)),
                'std_error': float(np.std(p1p2_errors)),
                'min_error': float(np.min(p1p2_errors)),
                'max_error': float(np.max(p1p2_errors)),
                'samples': len(p1p2_errors)
            }
        }

        # Calculate percentiles
        for model_name, errors in [('combined_model', combined_errors), ('p1p2_model', p1p2_errors)]:
            percentiles = np.percentile(errors, [50, 75, 90, 95, 99])
            stats[model_name]['percentiles'] = {
                'p50': float(percentiles[0]),
                'p75': float(percentiles[1]),
                'p90': float(percentiles[2]),
                'p95': float(percentiles[3]),
                'p99': float(percentiles[4])
            }

            # Error thresholds
            within_1px = np.sum(np.array(errors) <= 1)
            within_2px = np.sum(np.array(errors) <= 2)
            within_5px = np.sum(np.array(errors) <= 5)
            within_10px = np.sum(np.array(errors) <= 10)

            stats[model_name]['error_thresholds'] = {
                'within_1px': int(within_1px),
                'within_2px': int(within_2px),
                'within_5px': int(within_5px),
                'within_10px': int(within_10px),
                'pct_within_1px': float(100 * within_1px / len(errors)),
                'pct_within_2px': float(100 * within_2px / len(errors)),
                'pct_within_5px': float(100 * within_5px / len(errors)),
                'pct_within_10px': float(100 * within_10px / len(errors))
            }

        # Save comparison results
        comparison_results = {
            'comparison_info': {
                'combined_model_landmarks': 31,
                'p1p2_model_landmarks': 2,
                'test_samples': len(combined_errors),
                'image_size': 768,
                'device': str(self.device)
            },
            'statistics': stats,
            'performance_comparison': {
                'combined_vs_p1p2_mean_error_ratio': stats['combined_model']['mean_error'] / stats['p1p2_model']['mean_error'],
                'combined_better_by': stats['p1p2_model']['mean_error'] - stats['combined_model']['mean_error'],
                'combined_pct_improvement': 100 * (stats['p1p2_model']['mean_error'] - stats['combined_model']['mean_error']) / stats['p1p2_model']['mean_error']
            }
        }

        with open(f"{output_dir}/model_comparison_results.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)

        # Generate text report
        report = f"""
MODEL ACCURACY COMPARISON REPORT
{'='*50}

Test Configuration:
- Combined Model: 31 landmarks (29 anatomical + 2 calibration)
- P1/P2 Model: 2 landmarks (calibration only)
- Test Samples: {len(combined_errors)}
- Image Size: 768x768

{'='*50}
ACCURACY METRICS
{'='*50}

Combined Model (31 landmarks):
- Mean Error: {stats['combined_model']['mean_error']:.2f} px
- Median Error: {stats['combined_model']['median_error']:.2f} px
- Std Deviation: {stats['combined_model']['std_error']:.2f} px
- Min/Max Error: {stats['combined_model']['min_error']:.2f} / {stats['combined_model']['max_error']:.2f} px

P1/P2 Model (2 landmarks):
- Mean Error: {stats['p1p2_model']['mean_error']:.2f} px
- Median Error: {stats['p1p2_model']['median_error']:.2f} px
- Std Deviation: {stats['p1p2_model']['std_error']:.2f} px
- Min/Max Error: {stats['p1p2_model']['min_error']:.2f} / {stats['p1p2_model']['max_error']:.2f} px

{'='*50}
PERCENTILES
{'='*50}

Combined Model:
- 50th percentile: {stats['combined_model']['percentiles']['p50']:.2f} px
- 75th percentile: {stats['combined_model']['percentiles']['p75']:.2f} px
- 90th percentile: {stats['combined_model']['percentiles']['p90']:.2f} px
- 95th percentile: {stats['combined_model']['percentiles']['p95']:.2f} px
- 99th percentile: {stats['combined_model']['percentiles']['p99']:.2f} px

P1/P2 Model:
- 50th percentile: {stats['p1p2_model']['percentiles']['p50']:.2f} px
- 75th percentile: {stats['p1p2_model']['percentiles']['p75']:.2f} px
- 90th percentile: {stats['p1p2_model']['percentiles']['p90']:.2f} px
- 95th percentile: {stats['p1p2_model']['percentiles']['p95']:.2f} px
- 99th percentile: {stats['p1p2_model']['percentiles']['p99']:.2f} px

{'='*50}
ERROR THRESHOLDS
{'='*50}

Combined Model:
- Within 1px: {stats['combined_model']['error_thresholds']['within_1px']}/{len(combined_errors)} ({stats['combined_model']['error_thresholds']['pct_within_1px']:.1f}%)
- Within 2px: {stats['combined_model']['error_thresholds']['within_2px']}/{len(combined_errors)} ({stats['combined_model']['error_thresholds']['pct_within_2px']:.1f}%)
- Within 5px: {stats['combined_model']['error_thresholds']['within_5px']}/{len(combined_errors)} ({stats['combined_model']['error_thresholds']['pct_within_5px']:.1f}%)
- Within 10px: {stats['combined_model']['error_thresholds']['within_10px']}/{len(combined_errors)} ({stats['combined_model']['error_thresholds']['pct_within_10px']:.1f}%)

P1/P2 Model:
- Within 1px: {stats['p1p2_model']['error_thresholds']['within_1px']}/{len(p1p2_errors)} ({stats['p1p2_model']['error_thresholds']['pct_within_1px']:.1f}%)
- Within 2px: {stats['p1p2_model']['error_thresholds']['within_2px']}/{len(p1p2_errors)} ({stats['p1p2_model']['error_thresholds']['pct_within_2px']:.1f}%)
- Within 5px: {stats['p1p2_model']['error_thresholds']['within_5px']}/{len(p1p2_errors)} ({stats['p1p2_model']['error_thresholds']['pct_within_5px']:.1f}%)
- Within 10px: {stats['p1p2_model']['error_thresholds']['within_10px']}/{len(p1p2_errors)} ({stats['p1p2_model']['error_thresholds']['pct_within_10px']:.1f}%)

{'='*50}
PERFORMANCE COMPARISON
{'='*50}

Combined vs P1/P2 Model:
- Error Ratio: {comparison_results['performance_comparison']['combined_vs_p1p2_mean_error_ratio']:.2f}x
- Combined better by: {comparison_results['performance_comparison']['combined_better_by']:.2f} px
- Improvement: {comparison_results['performance_comparison']['combined_pct_improvement']:.1f}%

{'='*50}
CONCLUSION
{'='*50}

"""

        if stats['combined_model']['mean_error'] < stats['p1p2_model']['mean_error']:
            report += "Combined model performs BETTER than specialized P1/P2 model\n"
            report += "   - Unified architecture is more effective\n"
            report += "   - Single model handles all landmarks efficiently\n"
        else:
            report += "P1/P2 specialized model performs better for calibration points\n"
            report += "   - Consider keeping separate models for critical accuracy\n"
            report += "   - Or fine-tune combined model further\n"

        report += "\n" + "="*50

        with open(f"{output_dir}/model_comparison_report.txt", 'w') as f:
            f.write(report)

        print("Comparison report generated")
        print(f"Report saved to: {output_dir}/model_comparison_report.txt")
        print(f"JSON results saved to: {output_dir}/model_comparison_results.json")

        return comparison_results


def main():
    """Main comparison function"""
    print("MODEL ACCURACY COMPARISON")
    print("="*50)

    comparator = ModelComparator()

    # Load models
    comparator.load_models()

    # Create test dataset
    dataset = comparator.create_comparison_dataset(
        annotations_file='annotations_p1_p2.json',
        images_dir='Aariz/train/Cephalograms',
        num_samples=20
    )

    # Create output directory
    output_dir = 'model_comparison_results'
    os.makedirs(output_dir, exist_ok=True)

    # Test models
    combined_results = comparator.test_combined_model(dataset, output_dir)
    p1p2_results = comparator.test_p1p2_model(dataset, output_dir)

    # Generate comparison report
    comparison = comparator.generate_comparison_report(output_dir)

    # Print summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)

    if comparison:
        stats = comparison['statistics']
        perf = comparison['performance_comparison']

        print("Combined Model (31 landmarks):")
        print(".2f")
        print(".2f")
        print(".1f")
        print(".1f")
        print("\nP1/P2 Model (2 landmarks):")
        print(".2f")
        print(".2f")
        print(".1f")
        print(".1f")
        print("\nPerformance Comparison:")
        print(".2f")
        print(".2f")
        print(".1f")
        if perf['combined_pct_improvement'] > 0:
            print("Combined model is better!")
        else:
            print("P1/P2 model is better for calibration points")

    print(f"\nResults saved in: {output_dir}/")
    print("="*50)


if __name__ == '__main__':
    main()