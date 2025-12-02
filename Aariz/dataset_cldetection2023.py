"""
Dataset Loader for CLdetection2023 Cephalometric Landmark Detection
Converts COCO format to Aariz format
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


class CLdetection2023Dataset(Dataset):
    """
    Dataset class for CLdetection2023 cephalometric landmark detection
    Converts COCO format annotations to Aariz format
    """
    def __init__(self, dataset_folder_path, mode="TRAIN", 
                 image_size=(768, 768), use_heatmap=True, heatmap_sigma=3.0, 
                 augmentation=True, train_split=0.8, val_split=0.1):
        """
        Args:
            dataset_folder_path: Path to CLdetection2023 dataset folder (should contain generated_images/)
            mode: "TRAIN", "VALID", or "TEST"
            image_size: Target image size (height, width)
            use_heatmap: Whether to use heatmap representation or direct coordinates
            heatmap_sigma: Sigma for Gaussian heatmap generation
            augmentation: Whether to apply data augmentation (only for training)
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
        """
        self.dataset_folder_path = dataset_folder_path
        self.mode = mode.upper()
        self.image_size = image_size
        self.use_heatmap = use_heatmap
        self.heatmap_sigma = heatmap_sigma
        self.augmentation = augmentation and (mode.upper() == "TRAIN")
        
        # Paths
        self.images_dir = os.path.join(dataset_folder_path, "generated_images", "image")
        self.annotations_file = os.path.join(dataset_folder_path, "generated_images", "annotations", "all_gen.json")
        
        # Load COCO format annotations
        self.annotations = self._load_annotations()
        
        # Split dataset
        self.image_ids = self._split_dataset(train_split, val_split)
        
        # CLdetection2023 has 38 landmarks
        # Map to 29 landmarks used in Aariz (we'll use first 29 or map accordingly)
        self.landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        # CLdetection2023 has 38 landmarks, we'll use all 38
        self.num_landmarks = 38
        
        # Image transforms
        self.transform = self._get_transforms()
        
    def _load_annotations(self):
        """Load COCO format annotations"""
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        # Create mapping from image_id to annotation
        annotations_dict = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_dict:
                annotations_dict[image_id] = []
            annotations_dict[image_id].append(ann)
        
        # Create mapping from image_id to image info
        images_dict = {}
        for img in data['images']:
            images_dict[img['id']] = img
        
        # Create categories mapping (landmark names)
        categories_dict = {}
        for cat in data.get('categories', []):
            categories_dict[cat['id']] = cat['name']
        
        # Combine into a list of (image_id, image_info, annotations)
        annotations_list = []
        for image_id, image_info in images_dict.items():
            if image_id in annotations_dict:
                annotations_list.append({
                    'image_id': image_id,
                    'image_info': image_info,
                    'annotations': annotations_dict[image_id],
                    'categories': categories_dict
                })
        
        return annotations_list
    
    def _split_dataset(self, train_split, val_split):
        """Split dataset into train/val/test"""
        total = len(self.annotations)
        train_size = int(total * train_split)
        val_size = int(total * val_split)
        
        # Shuffle for random split
        indices = np.random.RandomState(42).permutation(total)
        
        if self.mode == "TRAIN":
            split_indices = indices[:train_size]
        elif self.mode == "VALID":
            split_indices = indices[train_size:train_size + val_size]
        else:  # TEST
            split_indices = indices[train_size + val_size:]
        
        return [self.annotations[i] for i in split_indices]
    
    def _get_transforms(self):
        """Get image transformation pipeline using Albumentations"""
        height, width = self.image_size
        
        if self.augmentation:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Resize(height, width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['landmark_names']))
        else:
            return A.Compose([
                A.Resize(height, width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['landmark_names']))
    
    def _parse_keypoints(self, annotation):
        """Parse keypoints from COCO format [x, y, visibility, ...] to list of (x, y)"""
        keypoints = annotation['keypoints']
        landmarks = []
        # CLdetection2023 has 38 landmarks, so keypoints should be 38 * 3 = 114 elements
        for i in range(0, len(keypoints), 3):
            if i + 2 < len(keypoints):
                x = float(keypoints[i])
                y = float(keypoints[i + 1])
                visibility = int(keypoints[i + 2])
                if visibility > 0 and x > 0 and y > 0:  # Valid and visible landmark
                    landmarks.append((x, y))
                else:
                    landmarks.append((0, 0))  # Invisible or invalid landmarks set to (0, 0)
            else:
                landmarks.append((0, 0))  # Missing data
        # Ensure we have exactly 38 landmarks
        while len(landmarks) < 38:
            landmarks.append((0, 0))
        return landmarks[:38]  # Take only first 38
    
    def _generate_heatmap(self, landmarks, height, width):
        """Generate Gaussian heatmaps for landmarks"""
        heatmaps = np.zeros((self.num_landmarks, height, width), dtype=np.float32)
        
        for i, (x, y) in enumerate(landmarks):
            if x > 0 and y > 0:  # Valid landmark
                # Create Gaussian heatmap
                y_coords, x_coords = np.ogrid[:height, :width]
                heatmap = np.exp(-((x_coords - x) ** 2 + (y_coords - y) ** 2) / (2 * self.heatmap_sigma ** 2))
                heatmaps[i] = heatmap
        
        return heatmaps
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        item = self.image_ids[idx]
        image_info = item['image_info']
        annotation = item['annotations'][0]  # Take first annotation (should be only one per image)
        
        # Load image
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Parse keypoints
        landmarks = self._parse_keypoints(annotation)
        
        # Store original landmarks for later use
        original_landmarks = landmarks.copy()
        
        # Convert to numpy array for albumentations (only valid landmarks)
        keypoints_for_transform = [(x, y) for x, y in landmarks if x > 0 and y > 0]
        landmark_names = [f"landmark_{i}" for i in range(len(keypoints_for_transform))]
        
        # Apply transforms
        if len(keypoints_for_transform) > 0:
            transformed = self.transform(image=image, keypoints=keypoints_for_transform, landmark_names=landmark_names)
            image = transformed['image']
            transformed_keypoints = transformed['keypoints']
        else:
            # No valid keypoints, just transform image
            transformed = self.transform(image=image)
            image = transformed['image']
            transformed_keypoints = []
        
        # Scale landmarks to new image size
        scale_x = self.image_size[1] / original_width
        scale_y = self.image_size[0] / original_height
        
        # Create full landmarks list with scaled coordinates
        full_landmarks = []
        keypoint_idx = 0
        for i, (x, y) in enumerate(original_landmarks):
            if x > 0 and y > 0:
                # Scale to new image size
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                full_landmarks.append((scaled_x, scaled_y))
            else:
                full_landmarks.append((0, 0))
        
        # Generate heatmaps or use coordinates
        if self.use_heatmap:
            height, width = self.image_size
            heatmaps = self._generate_heatmap(full_landmarks, height, width)
            target = torch.from_numpy(heatmaps).float()
        else:
            # Direct coordinates
            coords = np.array(full_landmarks, dtype=np.float32)
            target = torch.from_numpy(coords).float()
        
        return {
            'image': image,
            'target': target,
            'landmarks': full_landmarks,
            'image_id': image_info['id'],
            'file_name': image_info['file_name']
        }


def create_dataloaders(dataset_folder_path, batch_size=8, num_workers=4,
                       image_size=(768, 768), use_heatmap=True, 
                       train_split=0.8, val_split=0.1, test_split=0.1,
                       prefetch_factor=2):
    """
    Create dataloaders for CLdetection2023 dataset
    
    Args:
        dataset_folder_path: Path to CLdetection2023 dataset folder
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (height, width)
        use_heatmap: Whether to use heatmap representation
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        prefetch_factor: Prefetch factor for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure splits sum to 1.0
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        # Normalize
        train_split = train_split / total_split
        val_split = val_split / total_split
        test_split = test_split / total_split
    
    train_dataset = CLdetection2023Dataset(
        dataset_folder_path=dataset_folder_path,
        mode="TRAIN",
        image_size=image_size,
        use_heatmap=use_heatmap,
        augmentation=True,
        train_split=train_split,
        val_split=val_split
    )
    
    val_dataset = CLdetection2023Dataset(
        dataset_folder_path=dataset_folder_path,
        mode="VALID",
        image_size=image_size,
        use_heatmap=use_heatmap,
        augmentation=False,
        train_split=train_split,
        val_split=val_split
    )
    
    test_dataset = CLdetection2023Dataset(
        dataset_folder_path=dataset_folder_path,
        mode="TEST",
        image_size=image_size,
        use_heatmap=use_heatmap,
        augmentation=False,
        train_split=train_split,
        val_split=val_split
    )
    
    # prefetch_factor only works with num_workers > 0
    prefetch_kwargs = {}
    if num_workers > 0:
        prefetch_kwargs['prefetch_factor'] = prefetch_factor
        prefetch_kwargs['persistent_workers'] = True
    else:
        prefetch_kwargs['persistent_workers'] = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **prefetch_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **prefetch_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **prefetch_kwargs
    )
    
    return train_loader, val_loader, test_loader

