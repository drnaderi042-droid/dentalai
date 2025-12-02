"""
Utility functions for training and evaluation
"""
import torch
import numpy as np
import os
from scipy.ndimage import maximum_filter


def heatmap_to_coordinates(heatmaps, height, width, use_argmax_threshold=0.3):
    """
    Convert heatmaps to landmark coordinates
    Uses soft-argmax (weighted average) for sub-pixel accuracy
    Falls back to argmax if heatmap is too flat (early training)
    
    Args:
        heatmaps: numpy array of shape (num_landmarks, height, width)
        height: height of heatmap
        width: width of heatmap
        use_argmax_threshold: If max value < this, use argmax instead of soft-argmax
    
    Returns:
        coordinates: numpy array of shape (num_landmarks, 2) with (x, y) coordinates
    """
    num_landmarks = heatmaps.shape[0]
    coordinates = np.zeros((num_landmarks, 2), dtype=np.float32)
    
    for i in range(num_landmarks):
        heatmap = heatmaps[i]
        
        max_val = heatmap.max()
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # If heatmap is too flat (early training), use argmax
        if max_val < use_argmax_threshold:
            coordinates[i] = [max_pos[1], max_pos[0]]  # (x, y)
        else:
            # Normalize heatmap to probability distribution
            # Apply temperature scaling to make distribution sharper
            temperature = 2.0  # Make distribution sharper
            heatmap_scaled = np.power(heatmap, temperature)
            heatmap_flat = heatmap_scaled.flatten()
            heatmap_sum = heatmap_flat.sum()
            
            if heatmap_sum > 1e-6:  # Valid heatmap
                # Use soft-argmax for sub-pixel accuracy
                # Weighted average of positions (focusing on high-value regions)
                weights = heatmap_flat / heatmap_sum
                
                # Get positions
                y_coords = np.arange(height).repeat(width)
                x_coords = np.tile(np.arange(width), height)
                
                # Calculate weighted average
                x_coord = np.sum(x_coords * weights)
                y_coord = np.sum(y_coords * weights)
                
                coordinates[i] = [x_coord, y_coord]
            else:
                # Fallback to argmax
                coordinates[i] = [max_pos[1], max_pos[0]]  # (x, y)
    
    return coordinates


def calculate_mre(predictions, targets):
    """
    Calculate Mean Radial Error (MRE) in pixels
    
    Args:
        predictions: numpy array of shape (N, num_landmarks, 2)
        targets: numpy array of shape (N, num_landmarks, 2)
    
    Returns:
        MRE in pixels
    """
    # Calculate radial error for each landmark
    radial_errors = np.sqrt(np.sum((predictions - targets) ** 2, axis=2))  # (N, num_landmarks)
    
    # Filter out invalid landmarks (coordinates < 0)
    valid_mask = (predictions >= 0).all(axis=2) & (targets >= 0).all(axis=2)
    radial_errors[~valid_mask] = np.nan
    
    # Calculate mean radial error
    mre = np.nanmean(radial_errors)
    
    return mre


def calculate_sdr(predictions, targets, pixel_sizes, orig_sizes, current_size, threshold_mm=2.0):
    """
    Calculate Success Detection Rate (SDR) at a given threshold
    
    Args:
        predictions: numpy array of shape (N, num_landmarks, 2)
        targets: numpy array of shape (N, num_landmarks, 2)
        pixel_sizes: numpy array of shape (N,) - pixel size in mm for each image
        orig_sizes: numpy array of shape (N, 2) - original image sizes (height, width)
        current_size: tuple of (height, width) - current image size after resizing
        threshold_mm: threshold in millimeters
    
    Returns:
        SDR percentage
    """
    num_samples = len(predictions)
    num_landmarks = predictions.shape[1]
    
    success_count = 0
    total_count = 0
    
    for i in range(num_samples):
        pixel_size = pixel_sizes[i]
        orig_size = orig_sizes[i]
        current_h, current_w = current_size
        
        # Scale predictions and targets to original image size
        scale_x = orig_size[1] / current_w  # orig_width / current_width
        scale_y = orig_size[0] / current_h  # orig_height / current_height
        
        pred_scaled = predictions[i].copy()
        pred_scaled[:, 0] *= scale_x
        pred_scaled[:, 1] *= scale_y
        
        target_scaled = targets[i].copy()
        target_scaled[:, 0] *= scale_x
        target_scaled[:, 1] *= scale_y
        
        # Calculate radial error in millimeters for each landmark
        radial_errors = np.sqrt(np.sum((pred_scaled - target_scaled) ** 2, axis=1))
        radial_errors_mm = radial_errors * pixel_size
        
        # Count successful detections
        valid_mask = (predictions[i] >= 0).all(axis=1) & (targets[i] >= 0).all(axis=1)
        valid_errors = radial_errors_mm[valid_mask]
        
        success_count += np.sum(valid_errors <= threshold_mm)
        total_count += np.sum(valid_mask)
    
    if total_count == 0:
        return 0.0
    
    sdr = (success_count / total_count) * 100.0
    return sdr


def save_checkpoint(checkpoint, save_dir, is_best=False, epoch=None):
    """
    Save model checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save latest checkpoint
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save epoch checkpoint
    if epoch is not None:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model (MRE: {checkpoint['best_mre']:.2f}mm)")


def load_checkpoint(checkpoint_path, model, optimizer=None, strict=True):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: مسیر فایل checkpoint
        model: مدل PyTorch
        optimizer: optimizer (اختیاری)
        strict: اگر True باشد، همه کلیدها باید مطابقت داشته باشند (default: True)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # بارگذاری وزن‌های مدل
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except RuntimeError as e:
        if strict:
            # اگر strict=True بود و خطا داد، دوباره با strict=False امتحان کن
            print(f"⚠️  Warning: Strict loading failed, trying with strict=False: {e}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def visualize_landmarks(image, landmarks, save_path=None):
    """
    Visualize landmarks on image
    
    Args:
        image: numpy array of shape (H, W, 3) or PIL Image
        landmarks: numpy array of shape (num_landmarks, 2) with (x, y) coordinates
        save_path: path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    
    # Plot landmarks
    valid_landmarks = landmarks[landmarks[:, 0] >= 0]
    if len(valid_landmarks) > 0:
        plt.scatter(valid_landmarks[:, 0], valid_landmarks[:, 1], 
                   c='red', s=50, marker='x', linewidths=2)
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()


