"""
Dataset Loader for 3471833 Cephalometric Landmark Detection (19 landmarks)
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Dataset3471833(Dataset):
    """
    Dataset class for 3471833 cephalometric landmark detection (19 landmarks)
    Based on ISBI standard 19-landmark format
    """
    def __init__(self, dataset_folder_path, mode="TRAIN", annotation_type="400_senior",
                 image_size=(512, 512), use_heatmap=True, heatmap_sigma=3.0, 
                 augmentation=True):
        """
        Args:
            dataset_folder_path: Path to 3471833 dataset folder
            mode: "TRAIN", "VALID", or "TEST"
            annotation_type: "400_senior" or "400_junior"
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
        
        # 19 landmarks in ISBI standard order
        # Based on common cephalometric landmarks and angle.py calculations
        self.landmark_symbols = [
            "S",      # 0: Sella
            "N",      # 1: Nasion
            "Or",     # 2: Orbitale
            "A",      # 3: Subspinale
            "B",      # 4: Supramentale
            "PNS",    # 5: Posterior Nasal Spine
            "ANS",    # 6: Anterior Nasal Spine
            "U1",     # 7: Upper Incisor Tip
            "L1",     # 8: Lower Incisor Tip
            "Me",     # 9: Menton
            "U6",     # 10: Upper Molar
            "L6",     # 11: Lower Molar
            "Go",     # 12: Gonion
            "Pog",    # 13: Pogonion
            "Gn",     # 14: Gnathion
            "Ar",     # 15: Articulare
            "Co",     # 16: Condylion
            "Po",     # 17: Porion
            "R"       # 18: Ramus
        ]
        self.num_landmarks = len(self.landmark_symbols)
        
        # Get all image IDs
        self.image_ids = self._get_image_ids()
        
        # Image transforms
        self.transform = self._get_transforms()
    
    def _get_image_ids(self):
        """Get list of image IDs for current mode"""
        if self.mode == "TRAIN":
            images_folder = os.path.join(
                self.dataset_folder_path,
                "RawImage",
                "RawImage",
                "TrainingData"
            )
        elif self.mode == "VALID":
            # Use Test1Data for validation (151-300)
            images_folder = os.path.join(
                self.dataset_folder_path,
                "RawImage",
                "RawImage",
                "Test1Data"
            )
        else:  # TEST
            # Use Test2Data for test (301-400)
            images_folder = os.path.join(
                self.dataset_folder_path,
                "RawImage",
                "RawImage",
                "Test2Data"
            )
        
        if not os.path.exists(images_folder):
            return []
        
        # Get all image files
        image_extensions = ['*.bmp', '*.BMP', '*.png', '*.jpg', '*.jpeg']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        
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
    
    def _load_landmarks(self, annotation_path):
        """Load landmarks from TXT file"""
        landmarks = np.full((self.num_landmarks, 2), -1.0, dtype=np.float32)
        
        if not os.path.exists(annotation_path):
            return landmarks
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        # First 19 lines should be landmarks (x,y format)
        for i in range(min(19, len(lines))):
            line = lines[i].strip()
            if ',' in line:
                try:
                    x, y = map(float, line.split(','))
                    landmarks[i] = [x, y]
                except ValueError:
                    continue
        
        return landmarks
    
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
            "AnnotationsByMD",
            self.annotation_type,
            f"{image_id}.txt"
        )
        
        landmarks = self._load_landmarks(landmarks_path)
        
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
        
        # Default pixel size (can be adjusted if you have this info)
        # For ISBI dataset (3471833), typical pixel size is around 0.1-0.15 mm/pixel
        # Using 0.1 as default (same as Aariz dataset)
        # TODO: If you have actual pixel size information for 3471833 dataset, update this
        pixel_size = 0.1  # Default value - adjust if you know the actual pixel size
        
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
        """Find image path by ID"""
        if self.mode == "TRAIN":
            images_folder = os.path.join(
                self.dataset_folder_path,
                "RawImage",
                "RawImage",
                "TrainingData"
            )
        elif self.mode == "VALID":
            images_folder = os.path.join(
                self.dataset_folder_path,
                "RawImage",
                "RawImage",
                "Test1Data"
            )
        else:  # TEST
            images_folder = os.path.join(
                self.dataset_folder_path,
                "RawImage",
                "RawImage",
                "Test2Data"
            )
        
        for ext in ['.bmp', '.BMP', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_path = os.path.join(images_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        
        raise FileNotFoundError(f"Image not found for ID: {image_id}")


def create_dataloaders_3471833(dataset_folder_path, batch_size=8, num_workers=4,
                              image_size=(512, 512), use_heatmap=True, 
                              annotation_type="400_senior", heatmap_sigma=None,
                              pin_memory=True, prefetch_factor=2):
    """
    Create train, validation, and test dataloaders for 3471833 dataset
    """
    from torch.utils.data import DataLoader
    
    # Adaptive heatmap_sigma based on image size
    if heatmap_sigma is None:
        base_size = 512
        base_sigma = 3.0
        avg_size = (image_size[0] + image_size[1]) / 2
        heatmap_sigma = base_sigma * (avg_size / base_size)
        heatmap_sigma = max(0.75, heatmap_sigma)
    
    train_dataset = Dataset3471833(
        dataset_folder_path=dataset_folder_path,
        mode="TRAIN",
        annotation_type=annotation_type,
        image_size=image_size,
        use_heatmap=use_heatmap,
        heatmap_sigma=heatmap_sigma,
        augmentation=True
    )
    
    val_dataset = Dataset3471833(
        dataset_folder_path=dataset_folder_path,
        mode="VALID",
        annotation_type=annotation_type,
        image_size=image_size,
        use_heatmap=use_heatmap,
        heatmap_sigma=heatmap_sigma,
        augmentation=False
    )
    
    test_dataset = Dataset3471833(
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
        drop_last=True
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
    dataset_path = "../3471833"
    
    dataset = Dataset3471833(
        dataset_folder_path=dataset_path,
        mode="TRAIN",
        image_size=(512, 512),
        use_heatmap=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of landmarks: {dataset.num_landmarks}")
    print(f"Landmark symbols: {dataset.landmark_symbols}")
    
    # Test loading one sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Landmarks shape: {sample['landmarks'].shape}")

