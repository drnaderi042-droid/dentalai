"""
Advanced Loss Functions for Cephalometric Landmark Detection
- Dark-Pose Loss (from DarkPose paper)
- Wing Loss
- Combined Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DarkPoseLoss(nn.Module):
    """
    Dark-Pose Loss for heatmap regression
    Paper: Distribution-Aware Coordinate Representation for Human Pose Estimation
    Better handling of coordinate distribution in heatmaps
    """
    def __init__(self, beta=0.1, threshold=0.5):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps [B, C, H, W]
            target: Target heatmaps [B, C, H, W]
        """
        # Normalize predictions
        pred = torch.sigmoid(pred)
        
        # Calculate difference
        diff = (pred - target).abs()
        
        # Dark-Pose loss: focus on high-confidence regions
        # Weight by target values (higher weight for high-confidence regions)
        weights = torch.pow(target + self.beta, 2)
        
        # Calculate loss
        loss = weights * diff
        
        # Apply threshold for stability
        loss = torch.clamp(loss, max=self.threshold)
        
        return loss.mean()


class WingLoss(nn.Module):
    """
    Wing Loss for landmark detection
    Paper: Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks
    Better handling of small and large errors
    """
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps [B, C, H, W]
            target: Target heatmaps [B, C, H, W]
        """
        # Normalize predictions
        pred = torch.sigmoid(pred)
        
        # Calculate difference
        diff = (pred - target).abs()
        
        # Wing loss formula
        C = self.w - self.w * torch.log(1 + self.w / self.epsilon)
        
        # Split into two regions
        mask = diff < self.w
        
        # Region 1: small errors
        loss1 = self.w * torch.log(1 + diff[mask] / self.epsilon)
        
        # Region 2: large errors
        loss2 = diff[~mask] - C
        
        # Combine losses
        if mask.sum() > 0:
            if (~mask).sum() > 0:
                total_loss = (loss1.sum() + loss2.sum()) / (loss1.numel() + loss2.numel() + 1e-8)
            else:
                total_loss = loss1.mean()
        elif (~mask).sum() > 0:
            total_loss = loss2.mean()
        else:
            total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        return total_loss


class CombinedDarkWingLoss(nn.Module):
    """
    Combined Dark-Pose Loss + Wing Loss
    loss = 0.5 * DarkLoss() + 0.5 * WingLoss()
    """
    def __init__(self, dark_beta=0.1, dark_threshold=0.5, wing_w=10.0, wing_epsilon=2.0):
        super().__init__()
        self.dark_loss = DarkPoseLoss(beta=dark_beta, threshold=dark_threshold)
        self.wing_loss = WingLoss(w=wing_w, epsilon=wing_epsilon)
    
    def forward(self, pred, target):
        dark = self.dark_loss(pred, target)
        wing = self.wing_loss(pred, target)
        return 0.5 * dark + 0.5 * wing


class CVMLoss(nn.Module):
    """
    CVM (Cephalometric Vertical Measurement) Loss
    For multi-task learning - predicting vertical measurements
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_cvm, target_cvm):
        """
        Args:
            pred_cvm: Predicted CVM values [B, num_measurements]
            target_cvm: Target CVM values [B, num_measurements]
        """
        return self.mse_loss(pred_cvm, target_cvm)
















