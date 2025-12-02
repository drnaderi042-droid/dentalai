"""
Combined Dataset Loader for Fine-tuning
Combines Aariz dataset (29 landmarks) with 3471833 dataset (19 landmarks)
Maps 19 landmarks to corresponding 29 landmarks for fine-tuning
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from dataset import AarizDataset
from dataset_3471833 import Dataset3471833


class CombinedDataset(Dataset):
    """
    Combined dataset that uses both Aariz (29 landmarks) and 3471833 (19 landmarks)
    Maps 19 landmarks from 3471833 to corresponding positions in 29-landmark structure
    """
    def __init__(self, aariz_dataset=None, dataset_3471833=None, 
                 use_aariz_only=False, use_3471833_only=False):
        """
        Args:
            aariz_dataset: AarizDataset instance (29 landmarks)
            dataset_3471833: Dataset3471833 instance (19 landmarks)
            use_aariz_only: If True, only use Aariz dataset
            use_3471833_only: If True, only use 3471833 dataset
        """
        self.aariz_dataset = aariz_dataset
        self.dataset_3471833 = dataset_3471833
        self.use_aariz_only = use_aariz_only
        self.use_3471833_only = use_3471833_only
        
        # Mapping from 19-landmark indices to 29-landmark indices
        # Based on common landmark names
        self.landmark_mapping_19_to_29 = {
            0: 10,   # S -> S
            1: 4,    # N -> N
            2: 5,    # Or -> Or
            3: 0,    # A -> A
            4: 2,    # B -> B
            5: 7,    # PNS -> PNS
            6: 1,    # ANS -> ANS
            7: 20,   # U1 -> UPM (Upper Incisor, approximate)
            8: 23,   # L1 -> LIA (Lower Incisor, approximate)
            9: 13,   # Me -> Me
            10: 19,  # U6 -> UPM (Upper Molar, approximate)
            11: 22,  # L6 -> LMT (Lower Molar, approximate)
            12: 14,  # Go -> Go
            13: 6,   # Pog -> Pog
            14: 12,  # Gn -> Gn
            15: 11,  # Ar -> Ar
            16: 12,  # Co -> Co (mapped to Go as closest)
            17: 9,   # Po -> Po
            18: 8    # R -> R
        }
        
        # Calculate dataset lengths
        self.aariz_len = len(aariz_dataset) if aariz_dataset else 0
        self.dataset_3471833_len = len(dataset_3471833) if dataset_3471833 else 0
        
        if use_aariz_only:
            self.total_len = self.aariz_len
        elif use_3471833_only:
            self.total_len = self.dataset_3471833_len
        else:
            self.total_len = self.aariz_len + self.dataset_3471833_len
        
        # Number of landmarks in output (always 29 for compatibility)
        self.num_landmarks = 29
    
    def _map_19_to_29(self, landmarks_19):
        """
        Map 19 landmarks to 29-landmark structure
        """
        landmarks_29 = np.full((29, 2), -1.0, dtype=np.float32)
        
        for idx_19, idx_29 in self.landmark_mapping_19_to_29.items():
            if landmarks_19[idx_19][0] >= 0 and landmarks_19[idx_19][1] >= 0:
                landmarks_29[idx_29] = landmarks_19[idx_19]
        
        return landmarks_29
    
    def _map_19_heatmaps_to_29(self, heatmaps_19):
        """
        Map 19 heatmaps to 29-heatmap structure
        """
        heatmaps_29 = np.zeros((29, heatmaps_19.shape[1], heatmaps_19.shape[2]), 
                               dtype=np.float32)
        
        for idx_19, idx_29 in self.landmark_mapping_19_to_29.items():
            heatmaps_29[idx_29] = heatmaps_19[idx_19]
        
        return heatmaps_29
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if self.use_aariz_only:
            if self.aariz_dataset is None:
                raise ValueError("Aariz dataset is None but use_aariz_only is True")
            return self.aariz_dataset[idx]
        elif self.use_3471833_only:
            if self.dataset_3471833 is None:
                raise ValueError("3471833 dataset is None but use_3471833_only is True")
            sample = self.dataset_3471833[idx]
            # Map 19 landmarks to 29
            landmarks_29 = self._map_19_to_29(sample['landmarks'].numpy())
            sample['landmarks'] = torch.from_numpy(landmarks_29).float()
            
            # Map normalized landmarks
            landmarks_normalized_29 = self._map_19_to_29(sample['landmarks_normalized'].numpy())
            sample['landmarks_normalized'] = torch.from_numpy(landmarks_normalized_29).float()
            
            # Map heatmaps if using heatmaps
            if 'target' in sample and len(sample['target'].shape) == 3:
                heatmaps_29 = self._map_19_heatmaps_to_29(sample['target'].numpy())
                sample['target'] = torch.from_numpy(heatmaps_29).float()
            
            return sample
        else:
            # Combined: first aariz_len samples are from Aariz, rest from 3471833
            if idx < self.aariz_len:
                if self.aariz_dataset is None:
                    raise ValueError("Aariz dataset is None")
                return self.aariz_dataset[idx]
            else:
                if self.dataset_3471833 is None:
                    raise ValueError("3471833 dataset is None")
                sample = self.dataset_3471833[idx - self.aariz_len]
                # Map 19 landmarks to 29
                landmarks_29 = self._map_19_to_29(sample['landmarks'].numpy())
                sample['landmarks'] = torch.from_numpy(landmarks_29).float()
                
                # Map normalized landmarks
                landmarks_normalized_29 = self._map_19_to_29(sample['landmarks_normalized'].numpy())
                sample['landmarks_normalized'] = torch.from_numpy(landmarks_normalized_29).float()
                
                # Map heatmaps if using heatmaps
                if 'target' in sample and len(sample['target'].shape) == 3:
                    heatmaps_29 = self._map_19_heatmaps_to_29(sample['target'].numpy())
                    sample['target'] = torch.from_numpy(heatmaps_29).float()
                
                return sample


def create_combined_dataloaders(aariz_path, dataset_3471833_path,
                               batch_size=8, num_workers=4,
                               image_size=(512, 512), use_heatmap=True,
                               aariz_annotation_type="Senior Orthodontists",
                               dataset_3471833_annotation_type="400_senior",
                               heatmap_sigma=None,
                               use_aariz_only=False, use_3471833_only=False,
                               pin_memory=True, prefetch_factor=2):
    """
    Create combined dataloaders using both datasets
    
    Args:
        aariz_path: Path to Aariz dataset
        dataset_3471833_path: Path to 3471833 dataset
        use_aariz_only: If True, only use Aariz dataset
        use_3471833_only: If True, only use 3471833 dataset
        Other args: Same as create_dataloaders
    """
    from torch.utils.data import DataLoader
    
    # Adaptive heatmap_sigma
    if heatmap_sigma is None:
        base_size = 512
        base_sigma = 3.0
        avg_size = (image_size[0] + image_size[1]) / 2
        heatmap_sigma = base_sigma * (avg_size / base_size)
        heatmap_sigma = max(0.75, heatmap_sigma)
    
    # Create Aariz dataset
    aariz_train = None
    aariz_val = None
    aariz_test = None
    
    if not use_3471833_only:
        aariz_train = AarizDataset(
            dataset_folder_path=aariz_path,
            mode="TRAIN",
            annotation_type=aariz_annotation_type,
            image_size=image_size,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            augmentation=True
        )
        
        aariz_val = AarizDataset(
            dataset_folder_path=aariz_path,
            mode="VALID",
            annotation_type=aariz_annotation_type,
            image_size=image_size,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            augmentation=False
        )
        
        aariz_test = AarizDataset(
            dataset_folder_path=aariz_path,
            mode="TEST",
            annotation_type=aariz_annotation_type,
            image_size=image_size,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            augmentation=False
        )
    
    # Create 3471833 dataset
    dataset_3471833_train = None
    dataset_3471833_val = None
    dataset_3471833_test = None
    
    if not use_aariz_only:
        from dataset_3471833 import Dataset3471833
        
        dataset_3471833_train = Dataset3471833(
            dataset_folder_path=dataset_3471833_path,
            mode="TRAIN",
            annotation_type=dataset_3471833_annotation_type,
            image_size=image_size,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            augmentation=True
        )
        
        dataset_3471833_val = Dataset3471833(
            dataset_folder_path=dataset_3471833_path,
            mode="VALID",
            annotation_type=dataset_3471833_annotation_type,
            image_size=image_size,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            augmentation=False
        )
        
        dataset_3471833_test = Dataset3471833(
            dataset_folder_path=dataset_3471833_path,
            mode="TEST",
            annotation_type=dataset_3471833_annotation_type,
            image_size=image_size,
            use_heatmap=use_heatmap,
            heatmap_sigma=heatmap_sigma,
            augmentation=False
        )
    
    # Create combined datasets
    train_dataset = CombinedDataset(
        aariz_dataset=aariz_train,
        dataset_3471833=dataset_3471833_train,
        use_aariz_only=use_aariz_only,
        use_3471833_only=use_3471833_only
    )
    
    val_dataset = CombinedDataset(
        aariz_dataset=aariz_val,
        dataset_3471833=dataset_3471833_val,
        use_aariz_only=use_aariz_only,
        use_3471833_only=use_3471833_only
    )
    
    test_dataset = CombinedDataset(
        aariz_dataset=aariz_test,
        dataset_3471833=dataset_3471833_test,
        use_aariz_only=use_aariz_only,
        use_3471833_only=use_3471833_only
    )
    
    # Create dataloaders
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

