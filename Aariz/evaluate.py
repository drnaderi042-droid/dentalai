"""
Evaluation Script for Cephalometric Landmark Detection
"""
import os
# Disable albumentations update warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

from dataset import create_dataloaders
from model import get_model
from utils import (
    heatmap_to_coordinates,
    calculate_mre,
    calculate_sdr,
    load_checkpoint,
    visualize_landmarks
)


def evaluate(model, test_loader, device, save_predictions=False, output_dir='results'):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_image_ids = []
    all_pixel_sizes = []
    all_orig_sizes = []
    current_size = None
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            landmarks = batch['landmarks'].cpu().numpy()
            image_ids = batch['image_id']
            pixel_sizes = batch['pixel_size'].cpu().numpy()
            orig_sizes = batch['orig_size'].cpu().numpy()
            
            # Store current size from first batch
            if current_size is None:
                current_size = (images.shape[2], images.shape[3])
            
            # Forward pass
            outputs = model(images)
            
            # Resize outputs to match target size if needed
            if outputs.shape[2:] != targets.shape[2:]:
                outputs_resized = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            else:
                outputs_resized = outputs
            
            # Convert heatmaps to coordinates
            heatmaps = torch.sigmoid(outputs_resized).cpu().numpy()
            batch_size = heatmaps.shape[0]
            h, w = heatmaps.shape[2:]
            
            for i in range(batch_size):
                pred_coords = heatmap_to_coordinates(heatmaps[i], h, w)
                all_predictions.append(pred_coords)
                all_targets.append(landmarks[i])
                all_image_ids.append(image_ids[i])
                all_pixel_sizes.append(pixel_sizes[i])
                all_orig_sizes.append(orig_sizes[i])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_pixel_sizes = np.array(all_pixel_sizes)
    
    # Calculate metrics in pixels
    mre_pixels = calculate_mre(all_predictions, all_targets)
    
    # Calculate metrics in millimeters
    mre_mm_list = []
    std_mm_list = []
    
    for i in range(len(all_predictions)):
        pixel_size = all_pixel_sizes[i]
        orig_size = all_orig_sizes[i]
        
        # Scale predictions and targets to original image size
        scale_x = orig_size[1] / current_size[1]
        scale_y = orig_size[0] / current_size[0]
        
        pred_scaled = all_predictions[i].copy()
        pred_scaled[:, 0] *= scale_x
        pred_scaled[:, 1] *= scale_y
        
        target_scaled = all_targets[i].copy()
        target_scaled[:, 0] *= scale_x
        target_scaled[:, 1] *= scale_y
        
        # Calculate radial errors in millimeters
        radial_errors = np.sqrt(np.sum((pred_scaled - target_scaled) ** 2, axis=1))
        valid_mask = (all_predictions[i] >= 0).all(axis=1) & (all_targets[i] >= 0).all(axis=1)
        valid_errors = radial_errors[valid_mask] * pixel_size
        
        if len(valid_errors) > 0:
            mre_mm_list.append(np.mean(valid_errors))
            std_mm_list.append(np.std(valid_errors))
    
    mre_mm = np.mean(mre_mm_list)
    std_mm = np.mean(std_mm_list)
    
    # Calculate SDR at different thresholds
    sdr_2mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_size, threshold_mm=2.0)
    sdr_2_5mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_size, threshold_mm=2.5)
    sdr_3mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_size, threshold_mm=3.0)
    sdr_4mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_size, threshold_mm=4.0)
    
    # Per-landmark statistics
    landmark_errors = []
    for i in range(all_predictions.shape[1]):  # For each landmark
        landmark_errors_i = []
        for j in range(len(all_predictions)):  # For each sample
            if all_predictions[j, i, 0] >= 0 and all_targets[j, i, 0] >= 0:
                pixel_size = all_pixel_sizes[j]
                orig_size = all_orig_sizes[j]
                
                scale_x = orig_size[1] / images.shape[3]
                scale_y = orig_size[0] / images.shape[2]
                
                pred_scaled = all_predictions[j, i] * np.array([scale_x, scale_y])
                target_scaled = all_targets[j, i] * np.array([scale_x, scale_y])
                
                error_mm = np.sqrt(np.sum((pred_scaled - target_scaled) ** 2)) * pixel_size
                landmark_errors_i.append(error_mm)
        
        if len(landmark_errors_i) > 0:
            landmark_errors.append(np.mean(landmark_errors_i))
        else:
            landmark_errors.append(np.nan)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Mean Radial Error (MRE): {mre_mm:.2f} ± {std_mm:.2f} mm")
    print(f"MRE (pixels): {mre_pixels:.2f}")
    print(f"\nSuccess Detection Rate (SDR):")
    print(f"  2.0 mm: {sdr_2mm:.2f}%")
    print(f"  2.5 mm: {sdr_2_5mm:.2f}%")
    print(f"  3.0 mm: {sdr_3mm:.2f}%")
    print(f"  4.0 mm: {sdr_4mm:.2f}%")
    print("=" * 80)
    
    # Save results
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'mre_mm': float(mre_mm),
            'std_mm': float(std_mm),
            'mre_pixels': float(mre_pixels),
            'sdr_2mm': float(sdr_2mm),
            'sdr_2_5mm': float(sdr_2_5mm),
            'sdr_3mm': float(sdr_3mm),
            'sdr_4mm': float(sdr_4mm),
            'landmark_errors': [float(e) if not np.isnan(e) else None for e in landmark_errors]
        }
        
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    return {
        'mre_mm': mre_mm,
        'std_mm': std_mm,
        'mre_pixels': mre_pixels,
        'sdr_2mm': sdr_2mm,
        'sdr_2_5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm,
        'landmark_errors': landmark_errors,
        'predictions': all_predictions,
        'targets': all_targets,
        'image_ids': all_image_ids
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Cephalometric Landmark Detection Model')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'unet', 'hourglass', 'hrnet'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction results')
    parser.add_argument('--annotation_type', type=str, default='Senior Orthodontists',
                       choices=['Senior Orthodontists', 'Junior Orthodontists'],
                       help='Annotation type to use')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_landmarks=29)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Create test dataloader
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        dataset_folder_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_heatmap=True,
        annotation_type=args.annotation_type
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    results = evaluate(
        model, test_loader, device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

