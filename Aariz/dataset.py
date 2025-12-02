"""
Dataset Loader for Aariz Cephalometric Landmark Detection
"""
import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import glob
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AarizDataset(Dataset):
    """
    Dataset class for Aariz cephalometric landmark detection
    """
    def __init__(self, dataset_folder_path, mode="TRAIN", annotation_type="Senior Orthodontists",
                 image_size=(512, 512), use_heatmap=True, heatmap_sigma=3.0, 
                 augmentation=True):
        """
        Args:
            dataset_folder_path: Path to Aariz dataset folder
            mode: "TRAIN", "VALID", or "TEST"
            annotation_type: "Senior Orthodontists" or "Junior Orthodontists"
            image_size: Target image size (height, width)
            use_heatmap: Whether to use heatmap representation or direct coordinates
            heatmap_sigma: Sigma for Gaussian heatmap generation
            augmentation: Whether to apply data augmentation (only for training)
        """
        self.dataset_folder_path = dataset_folder_path
        self.mode = mode.upper()
        self.annotation_type = annotation_type
        self.image_size = image_size
        self.use_heatmap = use_heatmap
        self.heatmap_sigma = heatmap_sigma
        self.augmentation = augmentation and (mode.upper() == "TRAIN")
        
        # Load machine mappings for pixel size
        self.pixel_sizes = self._load_pixel_sizes()
        
        # Get all image IDs
        self.image_ids = self._get_image_ids()
        
        # Define 29 landmark symbols (standard order)
        self.landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        self.num_landmarks = len(self.landmark_symbols)
        
        # Image transforms
        self.transform = self._get_transforms()
        
    def _load_pixel_sizes(self):
        """Load pixel sizes from CSV file"""
        pixel_sizes = {}
        csv_path = os.path.join(self.dataset_folder_path, "cephalogram_machine_mappings.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                pixel_sizes[row['cephalogram_id']] = row['pixel_size']
        return pixel_sizes
    
    def _get_image_ids(self):
        """Get list of image IDs for current mode"""
        images_folder = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Cephalograms"
        )
        
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
        # Extract image IDs (filename without extension)
        image_ids = []
        for img_path in image_files:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            image_ids.append(img_id)
        
        return sorted(list(set(image_ids)))
    
    def _get_transforms(self):
        """Get image transformation pipeline using Albumentations"""
        height, width = self.image_size
        
        if self.augmentation:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.Resize(height=height, width=width),
                A.Normalize(mean=0.5, std=0.5),  # For [0,1] range
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            return A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _generate_heatmap(self, landmarks, orig_size):
        """Generate Gaussian heatmaps for landmarks"""
        h, w = orig_size
        heatmaps = np.zeros((self.num_landmarks, h, w), dtype=np.float32)
        
        for idx, (x, y) in enumerate(landmarks):
            if x >= 0 and y >= 0 and x < w and y < h:
                # Create coordinate grids
                x_coords = np.arange(w)
                y_coords = np.arange(h)
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Generate Gaussian heatmap
                heatmap = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * self.heatmap_sigma ** 2))
                heatmaps[idx] = heatmap
        
        return heatmaps
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image (original size)
        image_path = self._find_image_path(image_id)
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size  # (width, height)
        orig_size_hw = (orig_size[1], orig_size[0])  # (height, width)
        
        # Load landmarks (in original pixel coordinates)
        landmarks_path = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Annotations",
            "Cephalometric Landmarks",
            self.annotation_type,
            f"{image_id}.json"
        )
        
        with open(landmarks_path, 'r') as f:
            annotation = json.load(f)
        
        # Extract landmarks in standard order (pixel coords)
        landmark_dict = {lm['symbol']: (lm['value']['x'], lm['value']['y']) 
                        for lm in annotation['landmarks']}
        landmarks = np.array([landmark_dict.get(symbol, (-1, -1)) 
                             for symbol in self.landmark_symbols], dtype=np.float32)
        
        # Prepare keypoints for Albumentations (filter valid landmarks)
        keypoints = []
        valid_indices = []
        for i, (x, y) in enumerate(landmarks):
            if x >= 0 and y >= 0:
                keypoints.append([x, y])
                valid_indices.append(i)
        keypoints = np.array(keypoints)
        
        # Convert image to np.uint8 array
        image_array = np.array(image).astype(np.uint8)
        
        # Apply Albumentations transform
        transformed = self.transform(image=image_array, keypoints=keypoints)
        image_tensor = transformed['image']  # Already tensor and normalized
        transformed_keypoints = transformed['keypoints']  # In transformed space
        
        # Rebuild full landmarks array (transformed coords for valid, -1 for invalid)
        landmarks_transformed = np.full((self.num_landmarks, 2), -1.0, dtype=np.float32)
        for j, idx in enumerate(valid_indices):
            landmarks_transformed[idx] = transformed_keypoints[j]
        
        # Normalize landmarks to [0, 1] (using resized size)
        h, w = self.image_size
        landmarks_normalized = landmarks_transformed.copy()
        landmarks_normalized[:, 0] /= w  # x / width
        landmarks_normalized[:, 1] /= h  # y / height
        
        # For heatmaps, use transformed landmarks (already in resized space)
        if self.use_heatmap:
            heatmaps = self._generate_heatmap(landmarks_transformed, self.image_size)
            target = torch.from_numpy(heatmaps).float()
        else:
            # Direct coordinate regression
            target = torch.from_numpy(landmarks_transformed.flatten()).float()
        
        # Get pixel size if available
        pixel_size = self.pixel_sizes.get(image_id, 0.1)
        
        return {
            'image': image_tensor,
            'landmarks': torch.from_numpy(landmarks_transformed).float(),
            'landmarks_normalized': torch.from_numpy(landmarks_normalized).float(),
            'target': target,
            'image_id': image_id,
            'pixel_size': pixel_size,
            'orig_size': torch.tensor(orig_size_hw, dtype=torch.int32)  # Original (height, width)
        }
    
    def _find_image_path(self, image_id):
        """Find image path by ID (handle different extensions)"""
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


def create_dataloaders(dataset_folder_path, batch_size=8, num_workers=4,
                      image_size=(512, 512), use_heatmap=True, 
                      annotation_type="Senior Orthodontists", heatmap_sigma=None,
                      pin_memory=True, prefetch_factor=2):
    """
    Create train, validation, and test dataloaders
    """
    # Adaptive heatmap_sigma based on image size
    if heatmap_sigma is None:
        # Scale sigma proportionally to image size
        # For 512x512: sigma=3.0, for 128x128: sigma=0.75 (3.0 * 128/512)
        base_size = 512
        base_sigma = 3.0
        avg_size = (image_size[0] + image_size[1]) / 2
        heatmap_sigma = base_sigma * (avg_size / base_size)
        # Ensure minimum sigma
        heatmap_sigma = max(0.75, heatmap_sigma)
    
    train_dataset = AarizDataset(
        dataset_folder_path=dataset_folder_path,
        mode="TRAIN",
        annotation_type=annotation_type,
        image_size=image_size,
        use_heatmap=use_heatmap,
        heatmap_sigma=heatmap_sigma,
        augmentation=True
    )
    
    val_dataset = AarizDataset(
        dataset_folder_path=dataset_folder_path,
        mode="VALID",
        annotation_type=annotation_type,
        image_size=image_size,
        use_heatmap=use_heatmap,
        heatmap_sigma=heatmap_sigma,
        augmentation=False
    )
    
    test_dataset = AarizDataset(
        dataset_folder_path=dataset_folder_path,
        mode="TEST",
        annotation_type=annotation_type,
        image_size=image_size,
        use_heatmap=use_heatmap,
        heatmap_sigma=heatmap_sigma,
        augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    dataset_path = "Aariz"
    
    dataset = AarizDataset(
        dataset_folder_path=dataset_path,
        mode="TRAIN",
        image_size=(512, 512),
        use_heatmap=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of landmarks: {dataset.num_landmarks}")
    print(f"Landmark symbols: {dataset.landmark_symbols}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Landmarks shape: {sample['landmarks'].shape}")
