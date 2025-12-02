"""
Configuration file for training parameters
فایل تنظیمات برای پارامترهای آموزش
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """تنظیمات آموزش"""
    # Dataset
    dataset_path: str = "Aariz"
    annotation_type: str = "Senior Orthodontists"  # یا "Junior Orthodontists"
    
    # Model
    model_name: str = "hrnet"  # "resnet", "unet", "hourglass", "hrnet"
    num_landmarks: int = 29
    
    # Training
    batch_size: int = 6  # Optimized for RTX 3070 Ti with 512x512 + HRNet-w48 (larger than Hourglass)
    epochs: int = 250  # Paper used 250 epochs for best results
    learning_rate: float = 5e-4  # Paper uses 5e-4 for detection
    weight_decay: float = 1e-4
    
    # Image
    image_size: tuple = (512, 512)  # (height, width) - Increased from 256x256
    use_heatmap: bool = True
    heatmap_sigma: float = 3.5  # CRITICAL FIX: Was 1.0 - Must be 3-4 for proper Gaussian heatmaps
    
    # Loss
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    focal_weight: float = 0.5  # وزن focal loss در ترکیب با MSE
    
    # Augmentation
    augmentation: bool = True
    rotation_degrees: float = 5.0
    translate: tuple = (0.05, 0.05)  # (width, height)
    scale: tuple = (0.95, 1.05)
    brightness: float = 0.1
    contrast: float = 0.1
    
    # Optimizer
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "reduce_on_plateau", "cosine", "step"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # System
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 10  # ذخیره checkpoint هر N epoch
    
    # Resume
    resume: str = None  # مسیر checkpoint برای ادامه آموزش


# نمونه تنظیمات پیش‌فرض
DEFAULT_CONFIG = TrainingConfig()

# تنظیمات برای GPU با حافظه محدود
LOW_MEMORY_CONFIG = TrainingConfig(
    batch_size=4,
    image_size=(384, 384),
    heatmap_sigma=2.5,
    model_name="unet"
)

# تنظیمات برای دقت بالا
HIGH_ACCURACY_CONFIG = TrainingConfig(
    batch_size=6,
    image_size=(512, 512),
    heatmap_sigma=3.5,
    model_name="hourglass",
    epochs=150,
    learning_rate=5e-4
)
