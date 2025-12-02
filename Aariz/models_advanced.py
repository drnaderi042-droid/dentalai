"""
HRNet-W48 Model with 128x128 Heatmap Output
Multi-task: Landmark Detection + CVM Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import HRNetLandmarkModel, HRModule


class HRNetW48MultiTask(nn.Module):
    """
    HRNet-W48 with Multi-task Learning
    - Landmark Detection (29 landmarks)
    - CVM (Cephalometric Vertical Measurement) Prediction
    """
    def __init__(self, num_landmarks=29, num_cvm_measurements=10, heatmap_size=128):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_cvm_measurements = num_cvm_measurements
        self.heatmap_size = heatmap_size
        
        # Base HRNet-W48 (width=48)
        width = 48
        
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
        
        # Final layers for landmark heatmaps: Aggregate multi-resolution features
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
        
        # CVM prediction head (from highest resolution branch)
        self.cvm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(width, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_cvm_measurements)
        )
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels, width, num_modules, num_branches):
        """Create a stage with multiple modules"""
        modules = []
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
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize final heatmap layers
        for final_layer in self.final_layers:
            for module in final_layer:
                if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, -2.0)
    
    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            heatmaps: Landmark heatmaps [B, num_landmarks, heatmap_size, heatmap_size]
            cvm: CVM predictions [B, num_cvm_measurements]
        """
        # Store input size
        input_size = x.shape[2:]
        
        # Stem
        x = self.stem(x)
        
        # Stage 1
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
        
        # Sum all branches
        output = torch.zeros_like(output_list[0])
        for out in output_list:
            output = output + out
        
        # Upsample to heatmap_size (128x128)
        if output.shape[2:] != (self.heatmap_size, self.heatmap_size):
            output = F.interpolate(
                output, size=(self.heatmap_size, self.heatmap_size), 
                mode='bilinear', align_corners=False
            )
        
        # CVM prediction from highest resolution branch
        cvm = self.cvm_head(x_list[0])
        
        return output, cvm


class HRNetW32CropRefiner(nn.Module):
    """
    HRNet-W32 for Stage-2: Refining landmarks using 256x256 crops
    """
    def __init__(self, num_landmarks=29):
        super().__init__()
        self.num_landmarks = num_landmarks
        width = 32
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stage1 = self._make_stage(64, width, num_modules=1, num_branches=1)
        self.transition1 = self._make_transition([width], [width, width*2])
        self.stage2 = self._make_stage([width, width*2], width, num_modules=1, num_branches=2)
        self.transition2 = self._make_transition([width, width*2], [width, width*2, width*4])
        self.stage3 = self._make_stage([width, width*2, width*4], width, num_modules=2, num_branches=3)
        self.transition3 = self._make_transition([width, width*2, width*4], [width, width*2, width*4, width*8])
        self.stage4 = self._make_stage([width, width*2, width*4, width*8], width, num_modules=2, num_branches=4)
        
        # Final layers
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
        modules = []
        if isinstance(in_channels, int):
            in_channels_list = [in_channels]
        else:
            in_channels_list = in_channels
        
        for i in range(num_modules):
            modules.append(HRModule(in_channels_list, width, num_branches, i == 0))
            if i == 0:
                in_channels_list = [width * (2**j) for j in range(num_branches)]
        return nn.Sequential(*modules)
    
    def _make_transition(self, in_channels, out_channels):
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
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[-1], out_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels[i]),
                        nn.ReLU(inplace=True)
                    )
                )
        return nn.ModuleList(transition_layers)
    
    def _apply_transition(self, x_list, transition):
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
        x = self.stem(x)
        x = self.stage1(x)
        if not isinstance(x, list):
            x_list = [x]
        else:
            x_list = x
        
        x_list = self._apply_transition(x_list, self.transition1)
        x_list = self.stage2(x_list)
        x_list = self._apply_transition(x_list, self.transition2)
        x_list = self.stage3(x_list)
        x_list = self._apply_transition(x_list, self.transition3)
        x_list = self.stage4(x_list)
        
        output_list = []
        for i, x_branch in enumerate(x_list):
            output_list.append(self.final_layers[i](x_branch))
        
        target_size = output_list[0].shape[2:]
        for i in range(1, len(output_list)):
            if output_list[i].shape[2:] != target_size:
                output_list[i] = F.interpolate(
                    output_list[i], size=target_size, mode='bilinear', align_corners=False
                )
        
        output = torch.zeros_like(output_list[0])
        for out in output_list:
            output = output + out
        
        return output
















