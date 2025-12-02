"""
Optimized utility functions with additional optimizations
"""
import torch
import numpy as np
import os
import time
from scipy.ndimage import maximum_filter
from contextlib import contextmanager


def heatmap_to_coordinates(heatmaps, height, width):
    """
    Convert heatmaps to landmark coordinates
    Uses soft-argmax (weighted average) for sub-pixel accuracy
    """
    num_landmarks = heatmaps.shape[0]
    coordinates = np.zeros((num_landmarks, 2), dtype=np.float32)
    
    for i in range(num_landmarks):
        heatmap = heatmaps[i]
        heatmap_flat = heatmap.flatten()
        heatmap_sum = heatmap_flat.sum()
        
        if heatmap_sum > 1e-6:
            weights = heatmap_flat / heatmap_sum
            y_coords = np.arange(height).repeat(width)
            x_coords = np.tile(np.arange(width), height)
            x_coord = np.sum(x_coords * weights)
            y_coord = np.sum(y_coords * weights)
            coordinates[i] = [x_coord, y_coord]
        else:
            max_val = heatmap.max()
            if max_val > 0.01:
                max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
                coordinates[i] = [max_pos[1], max_pos[0]]
            else:
                coordinates[i] = [-1, -1]
    
    return coordinates


def heatmap_to_coordinates_vectorized(heatmaps, height, width):
    """
    Vectorized version for faster processing
    """
    num_landmarks = heatmaps.shape[0]
    coordinates = np.zeros((num_landmarks, 2), dtype=np.float32)
    
    # Create coordinate grids once
    y_coords = np.arange(height)[:, None].repeat(width, axis=1)
    x_coords = np.arange(width)[None, :].repeat(height, axis=0)
    
    for i in range(num_landmarks):
        heatmap = heatmaps[i]
        heatmap_sum = heatmap.sum()
        
        if heatmap_sum > 1e-6:
            weights = heatmap / heatmap_sum
            x_coord = np.sum(x_coords * weights)
            y_coord = np.sum(y_coords * weights)
            coordinates[i] = [x_coord, y_coord]
        else:
            max_val = heatmap.max()
            if max_val > 0.01:
                max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
                coordinates[i] = [max_pos[1], max_pos[0]]
            else:
                coordinates[i] = [-1, -1]
    
    return coordinates


def calculate_mre(predictions, targets):
    """Calculate Mean Radial Error (MRE) in pixels"""
    radial_errors = np.sqrt(np.sum((predictions - targets) ** 2, axis=2))
    valid_mask = (predictions >= 0).all(axis=2) & (targets >= 0).all(axis=2)
    radial_errors[~valid_mask] = np.nan
    mre = np.nanmean(radial_errors)
    return mre


def calculate_sdr(predictions, targets, pixel_sizes, orig_sizes, current_size, threshold_mm=2.0):
    """Calculate Success Detection Rate (SDR) at a given threshold"""
    num_samples = len(predictions)
    success_count = 0
    total_count = 0
    
    for i in range(num_samples):
        pixel_size = pixel_sizes[i]
        orig_size = orig_sizes[i]
        current_h, current_w = current_size
        
        scale_x = orig_size[1] / current_w
        scale_y = orig_size[0] / current_h
        
        pred_scaled = predictions[i].copy()
        pred_scaled[:, 0] *= scale_x
        pred_scaled[:, 1] *= scale_y
        
        target_scaled = targets[i].copy()
        target_scaled[:, 0] *= scale_x
        target_scaled[:, 1] *= scale_y
        
        radial_errors = np.sqrt(np.sum((pred_scaled - target_scaled) ** 2, axis=1))
        radial_errors_mm = radial_errors * pixel_size
        
        valid_mask = (predictions[i] >= 0).all(axis=1) & (targets[i] >= 0).all(axis=1)
        valid_errors = radial_errors_mm[valid_mask]
        
        success_count += np.sum(valid_errors <= threshold_mm)
        total_count += np.sum(valid_mask)
    
    if total_count == 0:
        return 0.0
    
    sdr = (success_count / total_count) * 100.0
    return sdr


def save_checkpoint(checkpoint, save_dir, is_best=False, epoch=None):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    if epoch is not None:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model (MRE: {checkpoint['best_mre']:.2f}mm)")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


@contextmanager
def benchmark_context(name="Operation", enabled=True):
    """Context manager for benchmarking code blocks"""
    if not enabled:
        yield
        return
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    print(f"{name}: {(end - start) * 1000:.2f}ms")


class PerformanceMonitor:
    """Monitor training performance metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.batch_times = []
        self.data_times = []
        self.forward_times = []
        self.backward_times = []
        self.start_time = None
        self.data_start = None
    
    def start_batch(self):
        self.start_time = time.time()
        self.data_start = time.time()
    
    def data_loaded(self):
        if self.data_start is not None:
            self.data_times.append(time.time() - self.data_start)
    
    def forward_done(self):
        if hasattr(self, '_forward_start'):
            self.forward_times.append(time.time() - self._forward_start)
    
    def backward_done(self):
        if hasattr(self, '_backward_start'):
            self.backward_times.append(time.time() - self._backward_start)
    
    def end_batch(self):
        if self.start_time is not None:
            self.batch_times.append(time.time() - self.start_time)
    
    def get_stats(self):
        """Get average statistics"""
        stats = {}
        if self.batch_times:
            stats['avg_batch_time'] = np.mean(self.batch_times) * 1000  # ms
            stats['samples_per_sec'] = 1.0 / np.mean(self.batch_times)
        if self.data_times:
            stats['avg_data_time'] = np.mean(self.data_times) * 1000  # ms
        if self.forward_times:
            stats['avg_forward_time'] = np.mean(self.forward_times) * 1000  # ms
        if self.backward_times:
            stats['avg_backward_time'] = np.mean(self.backward_times) * 1000  # ms
        return stats
    
    def print_stats(self, prefix=""):
        """Print performance statistics"""
        stats = self.get_stats()
        print(f"\n{prefix}Performance Stats:")
        for key, value in stats.items():
            if 'time' in key:
                print(f"  {key}: {value:.2f}ms")
            else:
                print(f"  {key}: {value:.2f}")


def estimate_training_time(num_epochs, steps_per_epoch, avg_batch_time):
    """Estimate total training time"""
    total_seconds = num_epochs * steps_per_epoch * avg_batch_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def get_gpu_memory_stats():
    """Get GPU memory usage statistics"""
    if not torch.cuda.is_available():
        return {}
    
    stats = {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
        'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
    }
    return stats


def print_gpu_memory():
    """Print GPU memory usage"""
    stats = get_gpu_memory_stats()
    if stats:
        print(f"GPU Memory: Allocated={stats['allocated']:.2f}GB, "
              f"Reserved={stats['reserved']:.2f}GB, "
              f"Max={stats['max_allocated']:.2f}GB")


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def visualize_landmarks(image, landmarks, save_path=None):
    """Visualize landmarks on image"""
    import matplotlib.pyplot as plt
    
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    
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