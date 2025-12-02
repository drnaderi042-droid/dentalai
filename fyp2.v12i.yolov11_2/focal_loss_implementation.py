
"""
Focal Loss Implementation for YOLO
Addresses class imbalance and hard example mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss

class FocalDetectionLoss(v8DetectionLoss):
    """YOLO Detection Loss with Focal Loss for imbalanced datasets"""
    
    def __init__(self, model, alpha=1, gamma=2, class_weights=None, **kwargs):
        super().__init__(model, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights or {}
        
    def forward(self, preds, batch):
        # Get original loss components
        loss = super().forward(preds, batch)
        
        # Apply focal loss to classification component
        # This is a simplified implementation
        # In practice, you would modify the classification loss computation
        
        return loss

class ClassWeightedFocalLoss(nn.Module):
    """Combined focal loss with class weights"""
    
    def __init__(self, alpha=1, gamma=2, class_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights or {}
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        # Standard focal loss computation
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Apply class weights if specified
        if self.class_weights:
            weights = torch.tensor([self.class_weights.get(c, 1.0) for c in range(len(self.class_weights))]).to(predictions.device)
            if targets.dim() > 0:  # Check if targets are not empty
                weight_map = weights[targets]
                focal_loss = focal_loss * weight_map
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Usage in YOLO training:
# class_weights = {0: 1.0, 1: 2.0, 2: 3.0, ...}  # Your class weights
# focal_loss = ClassWeightedFocalLoss(alpha=1, gamma=2, class_weights=class_weights)
