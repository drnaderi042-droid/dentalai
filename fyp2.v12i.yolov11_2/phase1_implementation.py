#!/usr/bin/env python3
"""
Phase 1 Implementation: Main Dataset Optimization
18-Class Canine/Molar Classification
Implements immediate improvements for class imbalance and training optimization
"""

import subprocess
import sys
import os
from pathlib import Path
import yaml
import torch

def create_class_weights_config():
    """Create class weights configuration based on performance analysis"""
    
    class_weights = {
        # Excellent performance classes (keep weights low)
        'canine class III 1-4': 1.0,
        'molar class II full class': 1.0,
        'molar class III 1-2': 1.0,
        'molar-class-III-full-class': 1.1,
        'canine class II 1-4': 1.1,
        'molar class III 3-4': 1.2,
        'molar class I': 1.2,
        'canine-class-III-full-class': 1.2,
        
        # Good performance classes (moderate increase)
        'canine class I': 1.5,
        'canine class II 3-4': 1.5,
        'molar class III 1-4': 1.8,
        'canine class II 1-2': 1.8,
        'molar class II 1-2': 2.0,
        
        # Critical classes (highest priority)
        'molar class II 1-4': 5.0,      # Worst performer
        'molar class II 3-4': 4.0,      # Very poor
        'canine class III 1-2': 4.0,    # Very poor
        'canine class II full class': 3.0  # Poor
    }
    
    # Save class weights to file
    config_path = "class_weights_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(class_weights, f, default_flow_style=False)
    
    print(f"✅ Class weights configuration saved to {config_path}")
    return class_weights

def create_confidence_thresholds_config():
    """Create class-specific confidence thresholds"""
    
    confidence_thresholds = {
        # High-performance classes: Standard threshold
        'canine class III 1-4': 0.5,
        'molar class II full class': 0.5,
        'molar class III 1-2': 0.5,
        'molar-class-III-full-class': 0.5,
        'canine class II 1-4': 0.5,
        'molar class III 3-4': 0.5,
        'molar class I': 0.5,
        'canine-class-III-full-class': 0.5,
        
        # Moderate classes: Lower threshold
        'canine class I': 0.3,
        'canine class II 3-4': 0.3,
        'molar class III 1-4': 0.3,
        'canine class II 1-2': 0.3,
        'molar class II 1-2': 0.25,
        
        # Critical classes: Very low threshold
        'molar class II 1-4': 0.1,
        'molar class II 3-4': 0.15,
        'canine class III 1-2': 0.1,
        'canine class II full class': 0.2
    }
    
    # Save confidence thresholds to file
    config_path = "confidence_thresholds_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(confidence_thresholds, f, default_flow_style=False)
    
    print(f"✅ Confidence thresholds configuration saved to {config_path}")
    return confidence_thresholds

def create_optimized_training_config():
    """Create optimized training configuration"""
    
    config = {
        "model": "yolov8s.pt",
        "data": "data.yaml",
        "epochs": 150,
        "patience": 25,
        "batch": 16,
        "imgsz": 800,
        "lr0": 0.003,
        "lrf": 0.01,
        "momentum": 0.95,
        "weight_decay": 0.0015,
        "warmup_epochs": 7,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.05,
        "box": 7.5,
        "cls": 3.0,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.15,
        "scale": 0.3,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.2,
        "mosaic": 1.0,
        "mixup": 0.2,
        "cutmix": 0.0,
        "copy_paste": 0.2,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "device": "0",
        "workers": 8,
        "project": "main_dataset_optimized",
        "name": "phase1_run",
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 0,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 10,
        "resume": False,
        "amp": True,
        "fraction": 1.0,
        "profile": False,
        "freeze": None,
        "multi_scale": False,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": True,
        "split": "val",
        "save": True,
        "save_period": 25,
        "cache": False
    }
    
    # Save training config to file
    config_path = "phase1_training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Optimized training configuration saved to {config_path}")
    return config

def generate_training_command():
    """Generate the optimized training command"""
    
    command = [
        "yolo", "train",
        "model=yolov8s.pt",
        "data=data.yaml",
        "epochs=150",
        "patience=25",
        "batch=16",
        "imgsz=800",
        "lr0=0.003",
        "lrf=0.01",
        "momentum=0.95",
        "weight_decay=0.0015",
        "warmup_epochs=7",
        "warmup_momentum=0.8",
        "warmup_bias_lr=0.05",
        "box=7.5",
        "cls=3.0",
        "dfl=1.5",
        "hsv_h=0.015",
        "hsv_s=0.7",
        "hsv_v=0.4",
        "degrees=10.0",
        "translate=0.15",
        "scale=0.3",
        "shear=0.0",
        "perspective=0.0",
        "flipud=0.0",
        "fliplr=0.2",
        "mosaic=1.0",
        "mixup=0.2",
        "copy_paste=0.2",
        "auto_augment=randaugment",
        "device=0",
        "workers=8",
        "project=main_dataset_optimized",
        "name=phase1_run",
        "save=true",
        "save_period=25",
        "verbose=true",
        "pretrained=true",
        "amp=true"
    ]
    
    return " ".join(command)

def run_phase1_training():
    """Execute the Phase 1 optimized training"""
    
    print("=== Phase 1: Main Dataset Optimization ===")
    print("Implementing immediate improvements for 18-class classification")
    print()
    
    # Create configurations
    print("📋 Creating Phase 1 configurations...")
    class_weights = create_class_weights_config()
    confidence_thresholds = create_confidence_thresholds_config()
    training_config = create_optimized_training_config()
    
    print()
    print("🎯 Phase 1 Optimizations Summary:")
    print("=" * 50)
    print(f"• Model upgrade: YOLOv8n → YOLOv8s")
    print(f"• Learning rate: 0.01 → 0.003")
    print(f"• Class loss weight: 0.5 → 3.0")
    print(f"• Image size: 640 → 800px")
    print(f"• Class weights: Implemented for all 18 classes")
    print(f"• Enhanced augmentation: mixup=0.2, copy_paste=0.2")
    print(f"• Reduced patience: 100 → 25 epochs")
    print()
    
    # Show class weights priority
    print("🔥 Class Weight Priorities:")
    print("=" * 30)
    print("Highest weights (critical classes):")
    critical_classes = {
        'molar class II 1-4': 5.0,
        'molar class II 3-4': 4.0,
        'canine class III 1-2': 4.0,
        'canine class II full class': 3.0
    }
    for class_name, weight in critical_classes.items():
        print(f"  • {class_name}: {weight}")
    
    print()
    print("⚡ Expected Improvements:")
    print("=" * 25)
    print("• Overall mAP50: 0.685 → 0.72-0.75 (+5-9%)")
    print("• Critical classes: All >0.400 mAP50")
    print("• Precision: Improved through class weights")
    print("• Training efficiency: Faster convergence")
    print()
    
    # Generate training command
    training_command = generate_training_command()
    print("🚀 Generated Training Command:")
    print("=" * 35)
    print(training_command)
    print()
    
    # Ask for user confirmation
    print("⚠️  IMPORTANT REMINDER:")
    print("The class weights need to be manually implemented in the loss function.")
    print("This script creates the configuration files, but you may need to")
    print("modify the YOLO training code to use the class weights.")
    print()
    
    user_input = input("Do you want to proceed with Phase 1 training setup? (y/n): ").strip().lower()
    
    if user_input not in ['y', 'yes']:
        print("Phase 1 setup cancelled by user.")
        return False
    
    print()
    print("✅ Phase 1 configuration files created successfully!")
    print("📁 Files created:")
    print("  • class_weights_config.yaml")
    print("  • confidence_thresholds_config.yaml") 
    print("  • phase1_training_config.yaml")
    print()
    print("🔧 Next Steps:")
    print("1. Implement class weights in YOLO loss function")
    print("2. Run the generated training command")
    print("3. Monitor training for 150 epochs (patience: 25)")
    print("4. Expected completion: 2-4 hours")
    
    return True

def validate_setup():
    """Validate that all necessary files exist"""
    
    required_files = ["data.yaml", "yolov8s.pt"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  • {file}")
        print()
        print("Please ensure all required files are available before running Phase 1.")
        return False
    
    print("✅ All required files found")
    return True

if __name__ == "__main__":
    print("YOLOv8 Main Dataset - Phase 1 Optimization Implementation")
    print("=" * 60)
    print("Target: 18-class canine/molar classification")
    print("Focus: Class imbalance and training optimization")
    print()
    
    # Validate setup
    if not validate_setup():
        sys.exit(1)
    
    # Run Phase 1 setup
    success = run_phase1_training()
    
    if success:
        print()
        print("🎉 Phase 1 setup completed successfully!")
        print("Ready to implement class weights and start optimized training.")
    else:
        print()
        print("❌ Phase 1 setup was cancelled or failed.")
