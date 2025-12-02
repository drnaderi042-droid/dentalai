"""
Dataset Loader for CVM Stage Classification
"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CVMDataset(Dataset):
    """
    Dataset class for CVM stage classification
    """
    def __init__(self, dataset_folder_path, mode="TRAIN", image_size=(768, 768), augmentation=True):
        """
        Args:
            dataset_folder_path: Path to Aariz dataset folder
            mode: "TRAIN", "VALID", or "TEST"
            image_size: Target image size (height, width)
            augmentation: Whether to apply data augmentation (only for training)
        """
        self.dataset_folder_path = dataset_folder_path
        self.mode = mode.upper()
        self.image_size = image_size
        self.augmentation = augmentation and (mode.upper() == "TRAIN")
        
        # Get all image IDs with CVM annotations
        self.image_ids = self._get_image_ids_with_cvm()
        
        # Image transforms
        self.transform = self._get_transforms()
        
    def _get_image_ids_with_cvm(self):
        """Get list of image IDs that have CVM annotations"""
        # Try both possible paths
        possible_paths = [
            os.path.join(self.dataset_folder_path, self.mode.lower(), "Annotations", "CVM Stages"),
            os.path.join(self.dataset_folder_path, "Aariz", self.mode.lower(), "Annotations", "CVM Stages")
        ]
        
        cvm_annotations_folder = None
        for path in possible_paths:
            if os.path.exists(path):
                cvm_annotations_folder = path
                break
        
        if cvm_annotations_folder is None:
            print(f"Warning: CVM annotations folder not found. Tried: {possible_paths}")
            return []
        
        # Get all JSON files
        json_files = glob.glob(os.path.join(cvm_annotations_folder, "*.json"))
        
        image_ids = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'cvm_stage' in data and 'value' in data['cvm_stage']:
                        ceph_id = data.get('ceph_id', os.path.splitext(os.path.basename(json_file))[0])
                        image_ids.append(ceph_id)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return sorted(list(set(image_ids)))
    
    def _get_transforms(self):
        """Get image transformation pipeline using Albumentations"""
        height, width = self.image_size
        
        if self.augmentation:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Resize(height=height, width=width, always_apply=True),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
                ToTensorV2(always_apply=True)
            ])
        else:
            return A.Compose([
                A.Resize(height=height, width=width, always_apply=True),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
                ToTensorV2(always_apply=True)
            ])
    
    def _load_cvm_stage(self, image_id):
        """Load CVM stage from annotation file"""
        # Try both possible paths
        possible_paths = [
            os.path.join(self.dataset_folder_path, self.mode.lower(), "Annotations", "CVM Stages"),
            os.path.join(self.dataset_folder_path, "Aariz", self.mode.lower(), "Annotations", "CVM Stages")
        ]
        
        json_file = None
        for base_path in possible_paths:
            test_path = os.path.join(base_path, f"{image_id}.json")
            if os.path.exists(test_path):
                json_file = test_path
                break
        
        if json_file is None:
            return None
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'cvm_stage' in data and 'value' in data['cvm_stage']:
                    stage = data['cvm_stage']['value']
                    # CVM stages are 1-6, convert to 0-5 for classification
                    return int(stage) - 1
        except Exception as e:
            print(f"Error loading CVM stage for {image_id}: {e}")
        
        return None
    
    def _load_image(self, image_id):
        """Load image by ID"""
        # Try both possible paths
        possible_paths = [
            os.path.join(self.dataset_folder_path, self.mode.lower(), "Cephalograms"),
            os.path.join(self.dataset_folder_path, "Aariz", self.mode.lower(), "Cephalograms")
        ]
        
        # Try different extensions
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
        for base_path in possible_paths:
            for ext in extensions:
                image_path = os.path.join(base_path, f"{image_id}{ext}")
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path).convert('RGB')
                        return np.array(image)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        continue
        
        return None
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image = self._load_image(image_id)
        if image is None:
            # Return a dummy image if loading fails
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        # Load CVM stage
        cvm_stage = self._load_cvm_stage(image_id)
        if cvm_stage is None:
            # Default to stage 0 if not found
            cvm_stage = 0
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        return {
            'image': image_tensor,
            'label': torch.tensor(cvm_stage, dtype=torch.long),
            'image_id': image_id
        }


def create_cvm_dataloaders(dataset_folder_path, image_size=(768, 768), batch_size=8, num_workers=4):
    """
    Create train and validation dataloaders for CVM classification
    
    Args:
        dataset_folder_path: Path to Aariz dataset folder
        image_size: Target image size (height, width)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    train_dataset = CVMDataset(
        dataset_folder_path=dataset_folder_path,
        mode="TRAIN",
        image_size=image_size,
        augmentation=True
    )
    
    val_dataset = CVMDataset(
        dataset_folder_path=dataset_folder_path,
        mode="VALID",
        image_size=image_size,
        augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

