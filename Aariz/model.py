"""
Model Architecture for Cephalometric Landmark Detection
Using Heatmap-based approach with Hourglass/UNet-like structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class LandmarkHeatmapModel(nn.Module):
    """
    Hourglass-style model for landmark heatmap prediction
    Based on stacked hourglass networks for pose estimation
    """
    def __init__(self, num_landmarks=29, input_channels=3, base_channels=64):
        super(LandmarkHeatmapModel, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res1 = self._make_residual_block(base_channels, base_channels)
        self.res2 = self._make_residual_block(base_channels, base_channels * 2, stride=2)
        self.res3 = self._make_residual_block(base_channels * 2, base_channels * 2)
        self.res4 = self._make_residual_block(base_channels * 2, base_channels * 4, stride=2)
        self.res5 = self._make_residual_block(base_channels * 4, base_channels * 4)
        
        # Hourglass modules (stacked)
        self.hg1 = HourglassModule(base_channels * 4, depth=4)
        self.hg2 = HourglassModule(base_channels * 4, depth=4)
        
        # Feature fusion
        self.features = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Output layers
        self.heatmap_conv = nn.Conv2d(base_channels * 2, num_landmarks, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        layers = []
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Hourglass modules
        x = self.hg1(x)
        x = self.hg2(x)
        
        # Feature fusion and output
        x = self.features(x)
        heatmaps = self.heatmap_conv(x)
        
        # Upsample to original resolution if needed
        # (assuming input was 512x512, this should give 128x128)
        # You may need to adjust based on your downsampling factor
        heatmaps = F.interpolate(heatmaps, size=(128, 128), mode='bilinear', align_corners=False)
        
        return heatmaps


class HourglassModule(nn.Module):
    """
    Single Hourglass module
    """
    def __init__(self, channels, depth=4):
        super(HourglassModule, self).__init__()
        self.depth = depth
        self.channels = channels
        
        if depth > 1:
            self.down_sample = nn.MaxPool2d(2, stride=2)
            self.low_branch = HourglassModule(channels, depth - 1)
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.low_branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        
        # Skip connections
        self.skip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        if self.depth > 1:
            # Downsample
            down = self.down_sample(x)
            # Lower branch
            low = self.low_branch(down)
            # Upsample
            up = self.up_sample(low)
            # Skip connection
            skip = self.skip(x)
            return up + skip
        else:
            return self.low_branch(x) + x


class UNetLandmarkModel(nn.Module):
    """
    UNet-based model for landmark heatmap prediction
    Simpler alternative to Hourglass
    """
    def __init__(self, num_landmarks=29, input_channels=3):
        super(UNetLandmarkModel, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._conv_block(1024 + 512, 512)
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)
        
        # Output
        self.output = nn.Conv2d(64, num_landmarks, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = F.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        heatmaps = self.output(dec1)
        
        return heatmaps


class ResNetLandmarkModel(nn.Module):
    """
    ResNet-based model with decoder for heatmap prediction
    Uses pretrained ResNet encoder
    """
    def __init__(self, num_landmarks=29, backbone='resnet50', pretrained=True):
        super(ResNetLandmarkModel, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Load pretrained ResNet
        if backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Encoder
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Get number of channels from backbone
        if '50' in backbone or '101' in backbone or '152' in backbone:
            encoder_channels = 2048
        else:
            encoder_channels = 512
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, kernel_size=1)
        )
        
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode to heatmaps
        heatmaps = self.decoder(features)
        
        # Ensure output matches input spatial dimensions
        if heatmaps.shape[2:] != x.shape[2:]:
            heatmaps = F.interpolate(heatmaps, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return heatmaps


class HRNetLandmarkModel(nn.Module):
    """
    HRNet (High-Resolution Network) for landmark detection
    Maintains high-resolution representations throughout the network
    Better than ResNet/UNet for pose estimation and landmark detection
    """
    def __init__(self, num_landmarks=29, width=32):
        super(HRNetLandmarkModel, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Stem (initial feature extraction)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: Create multi-resolution branches
        self.stage1 = self._make_stage(64, width, num_modules=1, num_branches=1)
        
        # Stage 2: Add second branch (2 resolutions)
        self.transition1 = self._make_transition([width], [width, width*2])
        self.stage2 = self._make_stage([width, width*2], width, num_modules=1, num_branches=2)
        
        # Stage 3: Add third branch (3 resolutions)
        self.transition2 = self._make_transition([width, width*2], [width, width*2, width*4])
        self.stage3 = self._make_stage([width, width*2, width*4], width, num_modules=2, num_branches=3)
        
        # Stage 4: Add fourth branch (4 resolutions)
        self.transition3 = self._make_transition([width, width*2, width*4], [width, width*2, width*4, width*8])
        self.stage4 = self._make_stage([width, width*2, width*4, width*8], width, num_modules=2, num_branches=4)
        
        # Final layers: Aggregate multi-resolution features
        self.final_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, num_landmarks, kernel_size=1, bias=True)
            ),
            nn.Sequential(
                nn.Conv2d(width*2, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(width, num_landmarks, kernel_size=1, bias=True)
            ),
            nn.Sequential(
                nn.Conv2d(width*4, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(width, num_landmarks, kernel_size=1, bias=True)
            ),
            nn.Sequential(
                nn.Conv2d(width*8, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
                nn.Conv2d(width, num_landmarks, kernel_size=1, bias=True)
            )
        ])
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels, width, num_modules, num_branches):
        """Create a stage with multiple modules"""
        modules = []
        # Convert to list if single int
        if isinstance(in_channels, int):
            in_channels_list = [in_channels]
        else:
            in_channels_list = in_channels
        
        for i in range(num_modules):
            modules.append(
                HRModule(in_channels_list, width, num_branches, i == 0)
            )
            if i == 0:
                in_channels_list = [width * (2**j) for j in range(num_branches)]
        return nn.Sequential(*modules)
    
    def _make_transition(self, in_channels, out_channels):
        """Create transition layer to add new branch"""
        num_branches_pre = len(in_channels)
        num_branches_cur = len(out_channels)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if out_channels[i] != in_channels[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # New branch: downsample from previous highest resolution
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[-1], out_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels[i]),
                        nn.ReLU(inplace=True)
                    )
                )
        return nn.ModuleList(transition_layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize final heatmap layers with smaller weights and bias
        # This helps produce reasonable initial heatmaps
        for final_layer in self.final_layers:
            for module in final_layer:
                if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
                    # This is the final heatmap output layer
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        # Bias towards low predictions initially
                        nn.init.constant_(module.bias, -2.0)
    
    def forward(self, x):
        # Store input size for final upsample
        input_size = x.shape[2:]
        
        # Stem
        x = self.stem(x)
        
        # Stage 1 (single branch)
        x = self.stage1(x)
        if not isinstance(x, list):
            x_list = [x]
        else:
            x_list = x
        
        # Stage 2
        x_list = self._apply_transition(x_list, self.transition1)
        x_list = self.stage2(x_list)
        
        # Stage 3
        x_list = self._apply_transition(x_list, self.transition2)
        x_list = self.stage3(x_list)
        
        # Stage 4
        x_list = self._apply_transition(x_list, self.transition3)
        x_list = self.stage4(x_list)
        
        # Final layers: upsample all branches to same resolution
        output_list = []
        for i, x_branch in enumerate(x_list):
            output_list.append(self.final_layers[i](x_branch))
        
        # Upsample all to highest resolution and sum
        target_size = output_list[0].shape[2:]
        for i in range(1, len(output_list)):
            if output_list[i].shape[2:] != target_size:
                output_list[i] = F.interpolate(
                    output_list[i], size=target_size, mode='bilinear', align_corners=False
                )
        
        # Sum all branches (create new tensor to avoid inplace)
        output = torch.zeros_like(output_list[0])
        for out in output_list:
            output = output + out
        
        # Upsample to input resolution if needed
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return output
    
    def _apply_transition(self, x_list, transition):
        """Apply transition layers"""
        x_list_output = []
        for i, trans in enumerate(transition):
            if trans is None:
                if i < len(x_list):
                    x_list_output.append(x_list[i])
                else:
                    x_list_output.append(x_list[-1])
            else:
                if i < len(x_list):
                    x_list_output.append(trans(x_list[i]))
                else:
                    x_list_output.append(trans(x_list[-1]))
        return x_list_output


class HRModule(nn.Module):
    """High-Resolution Module with multi-resolution branches and fusions"""
    def __init__(self, in_channels, width, num_branches, is_first):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        
        # Branches
        self.branches = nn.ModuleList()
        # Convert to list if single int
        if isinstance(in_channels, int):
            in_channels_list = [in_channels]
        else:
            in_channels_list = in_channels
        
        for i in range(num_branches):
            if is_first and i < len(in_channels_list):
                in_ch = in_channels_list[i]
            else:
                in_ch = width * (2 ** i)
            
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, width * (2 ** i), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width * (2 ** i)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width * (2 ** i), width * (2 ** i), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width * (2 ** i)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion layers
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)
                elif j > i:
                    # Upsample
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(width * (2 ** j), width * (2 ** i), kernel_size=1, bias=False),
                            nn.BatchNorm2d(width * (2 ** i))
                        )
                    )
                else:
                    # Downsample
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(width * (2 ** j), width * (2 ** i), kernel_size=3, 
                                     stride=2 ** (i - j), padding=1, bias=False),
                            nn.BatchNorm2d(width * (2 ** i))
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        
        # Forward through branches
        x_list = []
        for i, branch in enumerate(self.branches):
            if i < len(x):
                x_list.append(branch(x[i]))
            else:
                # New branch: downsample from previous
                x_list.append(branch(x[-1]))
        
        # Fusion
        x_fused = []
        for i in range(self.num_branches):
            y = x_list[i].clone()  # Clone to avoid inplace operations
            for j in range(self.num_branches):
                if i != j:
                    if j < len(x_list):
                        if self.fuse_layers[i][j] is not None:
                            if j > i:
                                # Upsample
                                scale = 2 ** (j - i)
                                y = y + F.interpolate(
                                    self.fuse_layers[i][j](x_list[j]),
                                    size=x_list[i].shape[2:],
                                    mode='bilinear',
                                    align_corners=False
                                )
                            else:
                                # Downsample
                                y = y + self.fuse_layers[i][j](x_list[j])
                        else:
                            y = y + x_list[j]
            x_fused.append(self.relu(y))
        
        return x_fused


class CephalometricLandmarkDetector(nn.Module):
    """
    Simple coordinate regression model for landmark detection.
    Directly outputs (x, y) coordinates instead of heatmaps.
    Perfect for small number of landmarks like p1/p2 calibration.
    """
    def __init__(self, num_landmarks=2, backbone='resnet18', pretrained=True):
        super(CephalometricLandmarkDetector, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Load pretrained ResNet backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
        else:
            resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        
        # Feature extractor (all layers except FC)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Regression head for coordinates
        # Output: num_landmarks * 2 (x, y for each landmark)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_landmarks * 2),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Initialize regression layers
        self._initialize_regression_layers()
    
    def _initialize_regression_layers(self):
        """Initialize regression head with smaller weights."""
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Regress coordinates
        coords = self.regressor(features)
        
        return coords


class HRNetP1P2Detector(nn.Module):
    """
    HRNet-based model for p1/p2 calibration point detection.
    Uses HRNet backbone with coordinate regression head.
    HRNet maintains high-resolution representations throughout the network.
    """
    def __init__(self, num_landmarks=2, hrnet_variant='hrnet_w18', pretrained=True):
        super(HRNetP1P2Detector, self).__init__()
        self.num_landmarks = num_landmarks
        
        try:
            import timm
            
            # Load pretrained HRNet from timm
            # Available variants: hrnet_w18, hrnet_w32, hrnet_w48
            if hrnet_variant not in ['hrnet_w18', 'hrnet_w32', 'hrnet_w48']:
                print(f"Warning: Unknown variant {hrnet_variant}, using hrnet_w18")
                hrnet_variant = 'hrnet_w18'
            
            self.backbone = timm.create_model(
                hrnet_variant,
                pretrained=pretrained,
                features_only=True,
                out_indices=[3]  # Get the highest resolution feature map
            )
            
            # Automatically detect feature dimension with a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_features = self.backbone(dummy_input)
                if isinstance(dummy_features, list):
                    dummy_features = dummy_features[-1]
                feature_dim = dummy_features.shape[1]  # Channel dimension
            
            print(f"Using {hrnet_variant} with auto-detected feature_dim={feature_dim}")
            
        except ImportError:
            print("timm library not found. Installing simple HRNet-like architecture...")
            # Fallback to ResNet if timm not available
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 512
            print("Using ResNet34 backbone as fallback (feature_dim=512)")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head for coordinates
        # Output: num_landmarks * 2 (x, y for each landmark)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_landmarks * 2),
            nn.Sigmoid()  # Output in [0, 1] range (normalized coordinates)
        )
        
        # Initialize regression layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize regression head with Xavier initialization."""
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extract features with HRNet backbone
        try:
            # timm models return a list of feature maps
            features = self.backbone(x)
            if isinstance(features, list):
                features = features[-1]  # Get the last feature map
        except:
            # Fallback backbone
            features = self.backbone(x)
        
        # Global average pooling
        features = self.global_pool(features)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Regress coordinates (normalized to [0, 1])
        coords = self.regressor(features)
        
        return coords


def get_model(model_name='resnet', num_landmarks=29, **kwargs):
    """
    Factory function to get model by name
    
    Available models:
    - hourglass: Stacked hourglass for heatmap prediction
    - unet: U-Net for heatmap prediction
    - resnet: ResNet with decoder for heatmap prediction
    - hrnet: HRNet (deprecated, use hrnet_p1p2 instead)
    - coordinate: Simple coordinate regression with ResNet
    - hrnet_p1p2: HRNet-based coordinate regression for p1/p2 detection
    """
    if model_name == 'hourglass':
        return LandmarkHeatmapModel(num_landmarks=num_landmarks, **kwargs)
    elif model_name == 'unet':
        return UNetLandmarkModel(num_landmarks=num_landmarks, **kwargs)
    elif model_name == 'resnet':
        return ResNetLandmarkModel(num_landmarks=num_landmarks, **kwargs)
    elif model_name == 'hrnet':
        # HRNet for 29 landmarks (heatmap-based)
        if num_landmarks == 2:
            # For p1/p2 detection, use HRNetP1P2Detector
            return HRNetP1P2Detector(num_landmarks=num_landmarks, **kwargs)
        else:
            # For 29 landmarks, use HRNetLandmarkModel
            return HRNetLandmarkModel(num_landmarks=num_landmarks, **kwargs)
    elif model_name == 'hrnet_p1p2':
        return HRNetP1P2Detector(num_landmarks=num_landmarks, **kwargs)
    elif model_name == 'coordinate':
        return CephalometricLandmarkDetector(num_landmarks=num_landmarks, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet, unet, hourglass, hrnet, hrnet_p1p2, coordinate")


if __name__ == "__main__":
    # Test model
    model = get_model('resnet', num_landmarks=29, backbone='resnet50')
    
    # Test input
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


