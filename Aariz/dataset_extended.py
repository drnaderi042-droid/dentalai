"""
Extended Dataset for Aariz with Additional Landmarks
Supports adding new landmarks like PT (Pterygoid) that are not in ISBI 2015 or CLdetection2023
"""
import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ExtendedAarizDataset(Dataset):
    """
    Extended Aariz dataset with additional landmarks
    Supports incremental addition of new landmarks
    """
    def __init__(self, dataset_folder_path, mode="TRAIN", annotation_type="Senior Orthodontists",
                 image_size=(512, 512), use_heatmap=True, heatmap_sigma=3.0, 
                 augmentation=True, additional_landmarks=None):
        """
        Args:
            dataset_folder_path: Path to Aariz dataset folder
            mode: "TRAIN", "VALID", or "TEST"
            annotation_type: "Senior Orthodontists" or "Junior Orthodontists"
            image_size: Target image size (height, width)
            use_heatmap: Whether to use heatmap representation
            heatmap_sigma: Sigma for Gaussian heatmap generation
            augmentation: Whether to apply data augmentation
            additional_landmarks: List of additional landmark symbols to add
                                 e.g., ["PT", "PTL", "PTR"] for Pterygoid landmarks
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
        
        # Base 29 landmark symbols (original Aariz landmarks)
        self.base_landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        
        # Additional landmarks (e.g., PT, PTL, PTR for Pterygoid)
        self.additional_landmarks = additional_landmarks if additional_landmarks else []
        
        # Combined landmark symbols
        self.landmark_symbols = self.base_landmark_symbols + self.additional_landmarks
        self.num_landmarks = len(self.landmark_symbols)
        
        # Create mapping from old indices to new indices
        self.old_to_new_mapping = {i: i for i in range(len(self.base_landmark_symbols))}
        
        # Image transforms
        self.transform = self._get_transforms()
        
        print(f"Extended dataset initialized:")
        print(f"  Base landmarks: {len(self.base_landmark_symbols)}")
        print(f"  Additional landmarks: {len(self.additional_landmarks)}")
        print(f"  Total landmarks: {self.num_landmarks}")
        if self.additional_landmarks:
            print(f"  Additional landmarks: {self.additional_landmarks}")
    
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
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
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
                A.Resize(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def _load_annotation(self, image_id):
        """Load annotation for an image"""
        annotation_path = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Annotations",
            "Cephalometric Landmarks",
            self.annotation_type,
            f"{image_id}.json"
        )
        
        if not os.path.exists(annotation_path):
            return None
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        return annotation
    
    def _generate_heatmap(self, landmarks, orig_size):
        """Generate Gaussian heatmaps for landmarks"""
        h, w = self.image_size
        heatmaps = np.zeros((self.num_landmarks, h, w), dtype=np.float32)
        
        # Scale landmarks to target size
        orig_h, orig_w = orig_size
        scale_x = w / orig_w
        scale_y = h / orig_h
        
        for i, landmark in enumerate(landmarks):
            if landmark[0] >= 0 and landmark[1] >= 0:  # Valid landmark
                x = int(landmark[0] * scale_x)
                y = int(landmark[1] * scale_y)
                
                # Create Gaussian heatmap
                if 0 <= x < w and 0 <= y < h:
                    y_coords, x_coords = np.ogrid[:h, :w]
                    dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
                    heatmaps[i] = np.exp(-dist_sq / (2 * self.heatmap_sigma ** 2))
        
        return heatmaps
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(
            self.dataset_folder_path,
            self.mode.lower(),
            "Cephalograms",
            f"{image_id}.png"
        )
        if not os.path.exists(image_path):
            image_path = image_path.replace('.png', '.jpg')
        
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size  # (width, height)
        
        # Load annotation
        annotation = self._load_annotation(image_id)
        
        # Extract landmarks
        landmarks_dict = {}
        if annotation:
            for lm in annotation.get('landmarks', []):
                symbol = lm['symbol']
                landmarks_dict[symbol] = {
                    'x': float(lm['value']['x']),
                    'y': float(lm['value']['y'])
                }
        
        # Create landmark array (base + additional)
        landmarks = np.full((self.num_landmarks, 2), -1.0, dtype=np.float32)
        
        # Fill base landmarks
        for i, symbol in enumerate(self.base_landmark_symbols):
            if symbol in landmarks_dict:
                landmarks[i] = [
                    landmarks_dict[symbol]['x'],
                    landmarks_dict[symbol]['y']
                ]
        
        # Fill additional landmarks (if available)
        for i, symbol in enumerate(self.additional_landmarks):
            idx = len(self.base_landmark_symbols) + i
            if symbol in landmarks_dict:
                landmarks[idx] = [
                    landmarks_dict[symbol]['x'],
                    landmarks_dict[symbol]['y']
                ]
        
        # Apply transforms
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # Generate heatmaps if needed
        if self.use_heatmap:
            # Scale landmarks for heatmap generation
            scale_x = self.image_size[1] / orig_size[0]
            scale_y = self.image_size[0] / orig_size[1]
            landmarks_scaled = landmarks.copy()
            landmarks_scaled[:, 0] *= scale_x
            landmarks_scaled[:, 1] *= scale_y
            
            heatmaps = self._generate_heatmap(landmarks_scaled, self.image_size)
            heatmaps = torch.from_numpy(heatmaps).float()
            return image_tensor, heatmaps, landmarks, image_id
        else:
            # Scale landmarks to target size
            scale_x = self.image_size[1] / orig_size[0]
            scale_y = self.image_size[0] / orig_size[1]
            landmarks_scaled = landmarks.copy()
            landmarks_scaled[:, 0] *= scale_x
            landmarks_scaled[:, 1] *= scale_y
            
            return image_tensor, landmarks_scaled, image_id


def create_extended_dataloaders(dataset_path, additional_landmarks=None, 
                                batch_size=4, num_workers=4, image_size=(512, 512)):
    """
    Create extended dataloaders for train, valid, and test sets
    
    Args:
        dataset_path: Path to Aariz dataset
        additional_landmarks: List of additional landmark symbols
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size (height, width)
    """
    train_dataset = ExtendedAarizDataset(
        dataset_path, mode="TRAIN", 
        image_size=image_size,
        augmentation=True,
        additional_landmarks=additional_landmarks
    )
    
    valid_dataset = ExtendedAarizDataset(
        dataset_path, mode="VALID",
        image_size=image_size,
        augmentation=False,
        additional_landmarks=additional_landmarks
    )
    
    test_dataset = ExtendedAarizDataset(
        dataset_path, mode="TEST",
        image_size=image_size,
        augmentation=False,
        additional_landmarks=additional_landmarks
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader, train_dataset.num_landmarks


if __name__ == "__main__":
    # Test the extended dataset
    dataset = ExtendedAarizDataset(
        "Aariz",
        mode="TRAIN",
        additional_landmarks=["PT", "PTL", "PTR"]  # Example: Pterygoid landmarks
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of landmarks: {dataset.num_landmarks}")
    print(f"Landmark symbols: {dataset.landmark_symbols}")
    
    # Test loading one sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample shape: {sample[0].shape if isinstance(sample[0], torch.Tensor) else 'N/A'}")
















