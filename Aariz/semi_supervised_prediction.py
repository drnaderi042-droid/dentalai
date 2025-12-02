"""
Semi-Supervised Learning Script for Adding New Landmarks
Uses existing model to predict new landmarks, then allows manual correction
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os
import glob
from tqdm import tqdm
from collections import defaultdict
import argparse

from model import get_model
from utils import load_checkpoint, heatmap_to_coordinates
from dataset_extended import ExtendedAarizDataset


def predict_with_uncertainty(model, image_tensor, device, num_samples=10):
    """
    Predict landmarks with uncertainty estimation using Monte Carlo Dropout
    """
    model.train()  # Enable dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(image_tensor.unsqueeze(0).to(device))
            predictions.append(pred.cpu().numpy())
    
    model.eval()
    
    predictions = np.array(predictions)  # [num_samples, 1, num_landmarks, H, W]
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred[0], std_pred[0]  # Remove batch dimension


def calculate_uncertainty_score(std_heatmap):
    """Calculate uncertainty score from standard deviation heatmap"""
    # Max uncertainty in the heatmap
    max_uncertainty = std_heatmap.max()
    # Mean uncertainty
    mean_uncertainty = std_heatmap.mean()
    # Weighted combination
    score = 0.7 * max_uncertainty + 0.3 * mean_uncertainty
    return score


def predict_new_landmarks(model, dataset_path, checkpoint_path, 
                         additional_landmarks, output_dir, 
                         uncertainty_threshold=0.5, device='cuda'):
    """
    Predict new landmarks for images without annotations
    Save predictions with uncertainty scores for manual review
    """
    # Load model
    model = get_model('hrnet', num_landmarks=29)
    checkpoint = load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()
    
    # Create extended dataset (without new landmarks in annotations)
    dataset = ExtendedAarizDataset(
        dataset_path,
        mode="TRAIN",
        additional_landmarks=additional_landmarks,
        augmentation=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Results storage
    results = []
    high_uncertainty_images = []
    
    print(f"Predicting landmarks for {len(dataset)} images...")
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image_tensor = sample[0]
        image_id = sample[-1] if len(sample) > 3 else f"image_{idx}"
        
        # Predict with uncertainty
        mean_pred, std_pred = predict_with_uncertainty(
            model, image_tensor, device, num_samples=10
        )
        
        # Extract only new landmarks (last N channels)
        num_new = len(additional_landmarks)
        new_mean = mean_pred[-num_new:]
        new_std = std_pred[-num_new:]
        
        # Convert to coordinates
        new_coords = []
        uncertainties = []
        
        for i in range(num_new):
            coords = heatmap_to_coordinates(new_mean[i:i+1])
            uncertainty = calculate_uncertainty_score(new_std[i])
            
            new_coords.append({
                'symbol': additional_landmarks[i],
                'x': float(coords[0][0]),
                'y': float(coords[0][1]),
                'uncertainty': float(uncertainty)
            })
            uncertainties.append(uncertainty)
        
        # Calculate overall uncertainty
        max_uncertainty = max(uncertainties)
        mean_uncertainty = np.mean(uncertainties)
        
        result = {
            'image_id': image_id,
            'landmarks': new_coords,
            'max_uncertainty': float(max_uncertainty),
            'mean_uncertainty': float(mean_uncertainty)
        }
        
        results.append(result)
        
        # Flag high uncertainty images
        if max_uncertainty > uncertainty_threshold:
            high_uncertainty_images.append(result)
    
    # Save results
    results_file = os.path.join(output_dir, 'predictions.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save high uncertainty images (priority for manual review)
    high_uncertainty_file = os.path.join(output_dir, 'high_uncertainty.json')
    with open(high_uncertainty_file, 'w', encoding='utf-8') as f:
        json.dump(high_uncertainty_images, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"High uncertainty images ({len(high_uncertainty_images)}): {high_uncertainty_file}")
    print(f"\nRecommendation: Review and correct high uncertainty images first")
    
    return results, high_uncertainty_images


def create_annotation_from_prediction(prediction, original_annotation_path, output_path):
    """
    Create annotation file from prediction
    Merge with existing annotations
    """
    # Load original annotation
    with open(original_annotation_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # Add new landmarks
    existing_symbols = {lm['symbol'] for lm in annotation.get('landmarks', [])}
    
    for new_lm in prediction['landmarks']:
        if new_lm['symbol'] not in existing_symbols:
            annotation['landmarks'].append({
                'symbol': new_lm['symbol'],
                'value': {
                    'x': new_lm['x'],
                    'y': new_lm['y']
                }
            })
    
    # Save updated annotation
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Semi-supervised prediction of new landmarks')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--additional_landmarks', type=str, nargs='+', default=['PT'],
                       help='Additional landmark symbols')
    parser.add_argument('--output_dir', type=str, default='predictions_new_landmarks',
                       help='Output directory for predictions')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.5,
                       help='Uncertainty threshold for flagging images')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Predict new landmarks
    results, high_uncertainty = predict_new_landmarks(
        None,  # Will be loaded inside function
        args.dataset_path,
        args.checkpoint,
        args.additional_landmarks,
        args.output_dir,
        args.uncertainty_threshold,
        device
    )
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("1. Review high uncertainty images in:", os.path.join(args.output_dir, 'high_uncertainty.json'))
    print("2. Correct predictions manually if needed")
    print("3. Use corrected annotations for fine-tuning")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
















