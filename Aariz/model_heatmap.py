"""
Heatmap-based P1/P2 detector - High accuracy for calibration points
Based on hrnet_p1p2_heatmap_best.pth approach
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available, using ResNet fallback")


class HRNetP1P2HeatmapDetector(nn.Module):
    """
    HRNet-based heatmap detector for P1/P2 calibration points.
    Outputs 2 heatmaps, then extracts coordinates from heatmaps.
    Target: < 10px error (similar to hrnet_p1p2_heatmap_best.pth)
    """
    def __init__(self, num_landmarks=2, hrnet_variant='hrnet_w18', pretrained=True, output_size=256):
        super(HRNetP1P2HeatmapDetector, self).__init__()
        self.num_landmarks = num_landmarks
        self.output_size = output_size  # Heatmap resolution (256x256 for 1024px input)

        if TIMM_AVAILABLE:
            # Load HRNet backbone
            self.backbone = timm.create_model(
                hrnet_variant,
                pretrained=pretrained,
                features_only=True,
                out_indices=[3]  # Use final stage features
            )

            # Get feature dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                if isinstance(features, list):
                    features = features[-1]
                feature_dim = features.shape[1]

            print(f"Using {hrnet_variant} with feature_dim={feature_dim}")
        else:
            # Fallback to ResNet
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 2048

        # Heatmap decoder for P1/P2 (2 landmarks)
        # Calculate required upsampling factor
        # Assuming HRNet outputs ~24x24 features for 1024x1024 input, we need to reach heatmap_size
        target_upsample = self.output_size // 24  # Approximate feature size from HRNet

        self.heatmap_decoder = nn.Sequential(
            # Initial processing
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Progressive upsampling to reach target size
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final upsampling to exact target size if needed
            nn.Upsample(size=(self.output_size, self.output_size), mode='bilinear', align_corners=False),

            # Generate heatmaps
            nn.Conv2d(32, num_landmarks, 1),  # Output: (batch, 2, H, W)
            nn.Sigmoid()  # Normalize to [0, 1] for heatmap probability
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass: extract features and generate heatmaps"""
        # Extract multi-scale features from HRNet
        features = self.backbone(x)
        if isinstance(features, list):
            features = features[-1]

        # Generate high-resolution heatmaps
        heatmaps = self.heatmap_decoder(features)

        return heatmaps

    def extract_coordinates(self, heatmaps, threshold=0.1):
        """
        Extract coordinates from heatmaps using advanced techniques
        Args:
            heatmaps: (batch, num_landmarks, H, W)
            threshold: Minimum confidence threshold
        Returns:
            coordinates: (batch, num_landmarks, 2) normalized coordinates
            confidences: (batch, num_landmarks) confidence scores
        """
        batch_size, num_landmarks, H, W = heatmaps.shape

        # Find peak locations for each landmark
        coordinates = []
        confidences = []

        for b in range(batch_size):
            batch_coords = []
            batch_confs = []

            for l in range(num_landmarks):
                heatmap = heatmaps[b, l]  # (H, W)

                # Apply threshold to remove noise
                heatmap_filtered = torch.where(heatmap > threshold, heatmap, torch.zeros_like(heatmap))

                if heatmap_filtered.sum() > 0:
                    # Find maximum location
                    max_val, max_idx = heatmap_filtered.view(-1).max(0)
                    max_y = max_idx // W
                    max_x = max_idx % W

                    # Sub-pixel refinement using quadratic interpolation
                    if max_y > 0 and max_y < H-1 and max_x > 0 and max_x < W-1:
                        # Get 3x3 neighborhood around peak
                        neighborhood = heatmap[max_y-1:max_y+2, max_x-1:max_x+2]

                        # Quadratic interpolation for sub-pixel accuracy
                        try:
                            # Simple center of mass for sub-pixel
                            y_coords = torch.arange(max_y-1, max_y+2, dtype=torch.float32, device=heatmap.device)
                            x_coords = torch.arange(max_x-1, max_x+2, dtype=torch.float32, device=heatmap.device)

                            y_sum = (neighborhood * y_coords.unsqueeze(1)).sum()
                            x_sum = (neighborhood * x_coords.unsqueeze(0)).sum()
                            total = neighborhood.sum()

                            if total > 0:
                                refined_y = y_sum / total
                                refined_x = x_sum / total
                            else:
                                refined_y = max_y.float()
                                refined_x = max_x.float()
                        except:
                            refined_y = max_y.float()
                            refined_x = max_x.float()
                    else:
                        refined_y = max_y.float()
                        refined_x = max_x.float()

                    # Normalize to [0, 1]
                    coord_x = (refined_x / (W - 1)).clamp(0, 1)
                    coord_y = (refined_y / (H - 1)).clamp(0, 1)

                    batch_coords.extend([coord_x.item(), coord_y.item()])
                    batch_confs.append(max_val.item())
                else:
                    # No confident detection, use center
                    batch_coords.extend([0.5, 0.5])
                    batch_confs.append(0.0)

            coordinates.append(batch_coords)
            confidences.append(batch_confs)

        return torch.tensor(coordinates, dtype=torch.float32, device=heatmaps.device), \
               torch.tensor(confidences, dtype=torch.float32, device=heatmaps.device)

    def extract_coordinates_batch(self, heatmaps):
        """
        Batch version of coordinate extraction for efficiency
        """
        return self.extract_coordinates(heatmaps)






