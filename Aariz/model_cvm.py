"""
CVM Stage Classification Model
Uses HRNet backbone with classification head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import HRNetLandmarkModel


class CVMClassifier(nn.Module):
    """
    CVM Stage Classifier using HRNet backbone
    Classifies cervical vertebral maturation stage (1-6)
    """
    def __init__(self, num_classes=6, backbone='hrnet', width=32, pretrained_backbone=None):
        """
        Args:
            num_classes: Number of CVM stages (6: stages 1-6)
            backbone: Backbone architecture ('hrnet' or 'resnet')
            width: Width parameter for HRNet
            pretrained_backbone: Path to pretrained backbone checkpoint (optional)
        """
        super(CVMClassifier, self).__init__()
        self.num_classes = num_classes
        
        if backbone == 'hrnet':
            # Use HRNet as backbone (without landmark head)
            self.backbone = HRNetLandmarkModel(num_landmarks=1, width=width)
            
            # Remove the final heatmap layers
            self.backbone.final_layers = nn.ModuleList()
            
            # Load pretrained weights if provided
            if pretrained_backbone:
                try:
                    checkpoint = torch.load(pretrained_backbone, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        # Filter out final_layers weights
                        backbone_state = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                        if not k.startswith('final_layers')}
                        self.backbone.load_state_dict(backbone_state, strict=False)
                        print(f"Loaded pretrained backbone from {pretrained_backbone}")
                except Exception as e:
                    print(f"Warning: Could not load pretrained backbone: {e}")
            
            # Global Average Pooling
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            # Classification head
            # HRNet width channels after stage 4 (use highest resolution branch)
            feature_dim = width
            
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            
            # Initialize classifier weights
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            # ResNet backbone
            import torchvision.models as models
            if backbone == 'resnet50':
                resnet = models.resnet50(weights='IMAGENET1K_V1')
                feature_dim = 2048
            elif backbone == 'resnet34':
                resnet = models.resnet34(weights='IMAGENET1K_V1')
                feature_dim = 512
            else:
                resnet = models.resnet18(weights='IMAGENET1K_V1')
                feature_dim = 512
            
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        # Extract features with backbone
        if isinstance(self.backbone, HRNetLandmarkModel):
            # HRNet forward
            input_size = x.shape[2:]
            x = self.backbone.stem(x)
            x = self.backbone.stage1(x)
            if not isinstance(x, list):
                x_list = [x]
            else:
                x_list = x
            
            x_list = self.backbone._apply_transition(x_list, self.backbone.transition1)
            x_list = self.backbone.stage2(x_list)
            x_list = self.backbone._apply_transition(x_list, self.backbone.transition2)
            x_list = self.backbone.stage3(x_list)
            x_list = self.backbone._apply_transition(x_list, self.backbone.transition3)
            x_list = self.backbone.stage4(x_list)
            
            # Use highest resolution branch for classification
            features = x_list[0]
        else:
            # ResNet forward
            features = self.backbone(x)
        
        # Global average pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def get_cvm_model(model_name='hrnet', num_classes=6, width=32, pretrained_backbone=None):
    """
    Get CVM classification model
    
    Args:
        model_name: Model architecture ('hrnet' or 'resnet18', 'resnet34', 'resnet50')
        num_classes: Number of CVM stages (default: 6)
        width: Width parameter for HRNet
        pretrained_backbone: Path to pretrained backbone checkpoint
    
    Returns:
        model: CVM classification model
    """
    if model_name == 'hrnet':
        return CVMClassifier(num_classes=num_classes, backbone='hrnet', width=width, 
                            pretrained_backbone=pretrained_backbone)
    elif model_name.startswith('resnet'):
        return CVMClassifier(num_classes=num_classes, backbone=model_name, 
                            pretrained_backbone=pretrained_backbone)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

