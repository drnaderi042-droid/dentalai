"""
Advanced Dataset with Multiple Augmentation Strategies
- Original: 1000 images
- Flipped: 1000 images (horizontal)
- Rotated ±7°: 2000 images
- Brightness: 1000 images
- Synthetic braces: 500 images
Total: 6500 images
"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import math


class AdvancedAarizDataset(Dataset):
    """
    Advanced dataset with multiple augmentation strategies
    """
    def __init__(self, dataset_folder_path, mode="TRAIN", annotation_type="Senior Orthodontists",
                 image_size=(512, 512), heatmap_size=128, heatmap_sigma=3.0,
                 augmentation_type="original"):
        """
        Args:
            dataset_folder_path: Path to Aariz dataset folder
            mode: "TRAIN", "VALID", or "TEST"
            annotation_type: "Senior Orthodontists" or "Junior Orthodontists"
            image_size: Target image size (height, width)
            heatmap_size: Target heatmap size (for 128x128 output)
            heatmap_sigma: Sigma for Gaussian heatmap generation
            augmentation_type: "original", "flipped", "rotated", "brightness", "synthetic_braces"
        """
        self.dataset_folder_path = dataset_folder_path
        self.mode = mode.upper()
        self.annotation_type = annotation_type
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.augmentation_type = augmentation_type
        
        # Load pixel sizes
        self.pixel_sizes = self._load_pixel_sizes()
        
        # Get image IDs
        self.image_ids = self._get_image_ids()
        
        # Limit to 1000 images for each augmentation type (except rotated which is 2000)
        if augmentation_type == "rotated":
            # Use each image twice (once for +7°, once for -7°)
            self.image_ids = self.image_ids[:1000] * 2
        else:
            self.image_ids = self.image_ids[:1000]
        
        # Define 29 landmark symbols
        self.landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        self.num_landmarks = len(self.landmark_symbols)
        
        # Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _load_pixel_sizes(self):
        """Load pixel size mappings"""
        pixel_sizes = {}
        csv_path = os.path.join(self.dataset_folder_path, "cephalogram_machine_mappings.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                pixel_sizes[row['cephalogram_id']] = row['pixel_size']
        return pixel_sizes
    
    def _get_image_ids(self):
        """Get all image IDs from dataset"""
        images_folder = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Cephalograms"
        )
        
        if not os.path.exists(images_folder):
            return []
        
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']:
            image_files.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
        
        image_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        return sorted(list(set(image_ids)))
    
    def _find_image_path(self, image_id):
        """Find image path by ID"""
        images_folder = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Cephalograms"
        )
        
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']:
            image_path = os.path.join(images_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        
        raise FileNotFoundError(f"Image not found for ID: {image_id}")
    
    def _generate_heatmap(self, landmarks, orig_size):
        """Generate Gaussian heatmaps for landmarks"""
        h, w = orig_size
        # Scale landmarks to heatmap size
        scale_h = self.heatmap_size / h
        scale_w = self.heatmap_size / w
        
        heatmaps = np.zeros((self.num_landmarks, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        for idx, (x, y) in enumerate(landmarks):
            if x >= 0 and y >= 0 and x < w and y < h:
                # Scale to heatmap coordinates
                x_hm = int(x * scale_w)
                y_hm = int(y * scale_h)
                
                x_hm = max(0, min(self.heatmap_size - 1, x_hm))
                y_hm = max(0, min(self.heatmap_size - 1, y_hm))
                
                # Create coordinate grids
                x_coords = np.arange(self.heatmap_size)
                y_coords = np.arange(self.heatmap_size)
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Generate Gaussian heatmap
                heatmap = np.exp(-((X - x_hm) ** 2 + (Y - y_hm) ** 2) / (2 * self.heatmap_sigma ** 2))
                heatmaps[idx] = heatmap
        
        return heatmaps
    
    def _apply_augmentation(self, image, landmarks, augmentation_type, index):
        """Apply specific augmentation"""
        if augmentation_type == "original":
            return image, landmarks
        
        elif augmentation_type == "flipped":
            # Horizontal flip
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = image.size
            landmarks_flipped = landmarks.copy()
            landmarks_flipped[:, 0] = w - landmarks_flipped[:, 0]
            return image, landmarks_flipped
        
        elif augmentation_type == "rotated":
            # Rotate ±7 degrees
            # Even indices: +7°, odd indices: -7°
            angle = 7.0 if index % 2 == 0 else -7.0
            
            w, h = image.size
            center = (w / 2, h / 2)
            
            # Rotate image
            image = image.rotate(angle, center=center, resample=Image.BICUBIC, fillcolor=0)
            
            # Rotate landmarks
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            landmarks_rotated = landmarks.copy()
            for i in range(len(landmarks_rotated)):
                x, y = landmarks_rotated[i]
                x -= center[0]
                y -= center[1]
                x_new = x * cos_a - y * sin_a
                y_new = x * sin_a + y * cos_a
                landmarks_rotated[i] = [x_new + center[0], y_new + center[1]]
            
            return image, landmarks_rotated
        
        elif augmentation_type == "brightness":
            # Adjust brightness
            brightness_factor = random.uniform(0.7, 1.3)
            
            # Convert to numpy for brightness adjustment
            img_array = np.array(image).astype(np.float32)
            img_array = img_array * brightness_factor
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(img_array)
            return image, landmarks
        
        elif augmentation_type == "synthetic_braces":
            # Add synthetic braces (simple overlay)
            # This is a simplified version - you can make it more sophisticated
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Add some bright lines to simulate braces
            # This is a placeholder - you can implement more realistic braces
            overlay = img_array.copy()
            
            # Add horizontal lines in the tooth region (middle third of image)
            y_start = int(h * 0.4)
            y_end = int(h * 0.6)
            
            for y in range(y_start, y_end, 20):
                cv2.line(overlay, (int(w * 0.3), y), (int(w * 0.7), y), (255, 255, 255), 2)
            
            # Blend with original
            alpha = 0.3
            img_array = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)
            image = Image.fromarray(img_array)
            
            return image, landmarks
        
        else:
            return image, landmarks
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self._find_image_path(image_id)
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size  # (width, height)
        orig_size_hw = (orig_size[1], orig_size[0])  # (height, width)
        
        # Load landmarks
        landmarks_path = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Annotations",
            "Cephalometric Landmarks",
            self.annotation_type,
            f"{image_id}.json"
        )
        
        with open(landmarks_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # Extract landmarks
        landmark_dict = {lm['symbol']: (lm['value']['x'], lm['value']['y']) 
                        for lm in annotation['landmarks']}
        landmarks = np.array([landmark_dict.get(symbol, (-1, -1)) 
                             for symbol in self.landmark_symbols], dtype=np.float32)
        
        # Apply augmentation
        image, landmarks = self._apply_augmentation(image, landmarks, self.augmentation_type, idx)
        
        # Resize image
        image = image.resize(self.image_size, Image.LANCZOS)
        
        # Scale landmarks to new size
        scale_x = self.image_size[1] / orig_size[0]
        scale_y = self.image_size[0] / orig_size[1]
        landmarks_scaled = landmarks.copy()
        landmarks_scaled[:, 0] *= scale_x
        landmarks_scaled[:, 1] *= scale_y
        
        # Generate heatmaps
        heatmaps = self._generate_heatmap(landmarks_scaled, self.image_size)
        
        # Convert to tensor
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)
        
        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        
        # CVM values (placeholder - you should calculate actual CVM from landmarks)
        cvm_values = self._calculate_cvm(landmarks_scaled)
        
        return {
            'image': image_tensor,
            'heatmaps': heatmaps_tensor,
            'landmarks': torch.from_numpy(landmarks_scaled).float(),
            'image_id': image_id,
            'pixel_size': self.pixel_sizes.get(image_id, 0.1),
            'orig_size': torch.tensor(orig_size_hw, dtype=torch.int32),
            'cvm': torch.from_numpy(cvm_values).float()
        }
    
    def _calculate_cvm(self, landmarks):
        """Calculate CVM (Cephalometric Vertical Measurement) values"""
        # Placeholder - implement actual CVM calculation
        # CVM typically includes measurements like:
        # - S-N length
        # - N-Me length
        # - etc.
        num_measurements = 10
        cvm_values = np.zeros(num_measurements, dtype=np.float32)
        
        # Example: calculate some basic measurements
        if len(landmarks) >= 29:
            # S-N distance
            if landmarks[10][0] >= 0 and landmarks[4][0] >= 0:  # S and N
                cvm_values[0] = np.sqrt(
                    (landmarks[10][0] - landmarks[4][0])**2 + 
                    (landmarks[10][1] - landmarks[4][1])**2
                )
            
            # N-Me distance
            if landmarks[4][0] >= 0 and landmarks[3][0] >= 0:  # N and Me
                cvm_values[1] = np.sqrt(
                    (landmarks[4][0] - landmarks[3][0])**2 + 
                    (landmarks[4][1] - landmarks[3][1])**2
                )
            
            # Add more measurements as needed
        
        return cvm_values
    
    def __len__(self):
        return len(self.image_ids)


def create_advanced_dataloaders(dataset_folder_path, batch_size=4, num_workers=4,
                                image_size=(512, 512), heatmap_size=128, heatmap_sigma=3.0,
                                annotation_type="Senior Orthodontists", pin_memory=True):
    """
    Create dataloaders with all augmentation types
    """
    augmentation_types = ["original", "flipped", "rotated", "brightness", "synthetic_braces"]
    
    datasets = []
    for aug_type in augmentation_types:
        dataset = AdvancedAarizDataset(
            dataset_folder_path=dataset_folder_path,
            mode="TRAIN",
            annotation_type=annotation_type,
            image_size=image_size,
            heatmap_size=heatmap_size,
            heatmap_sigma=heatmap_sigma,
            augmentation_type=aug_type
        )
        datasets.append(dataset)
    
    # Combine all datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(datasets)
    
    # Create dataloader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    # Validation and test loaders (original only)
    val_dataset = AdvancedAarizDataset(
        dataset_folder_path=dataset_folder_path,
        mode="VALID",
        annotation_type=annotation_type,
        image_size=image_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        augmentation_type="original"
    )
    
    test_dataset = AdvancedAarizDataset(
        dataset_folder_path=dataset_folder_path,
        mode="TEST",
        annotation_type=annotation_type,
        image_size=image_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        augmentation_type="original"
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader
















