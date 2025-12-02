#!/usr/bin/env python3
"""
Optimized Training Script for YOLOv11 Lateral Orthodontic AI
This script implements the recommended improvements to address class imbalance
and enhance model performance.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_optimized_training():
    """
    Execute the optimized training command with all recommended improvements
    """
    
    print("=== Starting Optimized YOLOv11 Training ===")
    print("Implementing recommendations from comprehensive model analysis")
    print()
    
    # Define training command with optimized parameters
    training_command = [
        "yolo", "train",
        "model=yolo11s.pt",  # Upgrade from nano to small
        "data=LATERAL ORTHO AI.v2i.yolov11/data.yaml",
        "epochs=150",
        "patience=20",  # Reduced from 100
        "batch=16",
        "imgsz=800",  # Increased for medical detail
        "lr0=0.005",  # Reduced learning rate
        "lrf=0.01",
        "momentum=0.95",
        "weight_decay=0.001",
        "warmup_epochs=5",
        "warmup_momentum=0.8",
        "warmup_bias_lr=0.05",
        "box=7.5",
        "cls=2.0",  # Increased class loss weight
        "dfl=1.5",
        "hsv_h=0.015",
        "hsv_s=0.7",
        "hsv_v=0.4",
        "degrees=5.0",
        "translate=0.1",
        "scale=0.2",
        "shear=0.0",
        "perspective=0.0",
        "flipud=0.0",
        "fliplr=0.3",  # Reduced from 0.5
        "mosaic=1.0",
        "mixup=0.1",  # Added
        "copy_paste=0.1",  # Added
        "auto_augment=randaugment",
        "device=0",
        "workers=8",
        "project=optimized_train",
        "name=run1",
        "exist_ok=false",
        "pretrained=true",
        "amp=true",
        "save=true",
        "save_period=25",
        "verbose=true"
    ]
    
    # Display the training configuration
    print("Training Configuration:")
    print("=====================")
    print(f"Model: YOLO11s (upgraded from YOLO11n)")
    print(f"Epochs: 150 (patience: 20)")
    print(f"Image Size: 800px (increased from 640px)")
    print(f"Learning Rate: 0.005 (reduced from 0.01)")
    print(f"Batch Size: 16")
    print(f"Class Loss Weight: 2.0 (increased from 0.5)")
    print()
    
    # Print expected improvements
    print("Expected Improvements:")
    print("=====================")
    print("• mAP50: 0.428 → 0.55-0.65")
    print("• mAP50-95: 0.206 → 0.30-0.40")
    print("• Better performance on minority classes")
    print("• Reduced class imbalance impact")
    print("• More stable training convergence")
    print()
    
    # Warn about class imbalance
    print("⚠️  CRITICAL REMINDER:")
    print("Your dataset has severe class imbalance (37.43:1 ratio)")
    print("Consider implementing class weights in the loss function:")
    print("  - Retroclination: weight 5.0")
    print("  - Open bite: weight 4.0")
    print("  - Crowding: weight 3.5")
    print("  - Cross bite: weight 3.0")
    print("  - Class II: weight 2.5")
    print()
    
    # Ask for user confirmation
    user_input = input("Do you want to proceed with the optimized training? (y/n): ").strip().lower()
    
    if user_input not in ['y', 'yes']:
        print("Training cancelled by user.")
        return False
    
    print()
    print("Starting training... This may take several hours.")
    print("=" * 60)
    
    try:
        # Execute training command
        result = subprocess.run(training_command, check=True, capture_output=False, text=True)
        print()
        print("✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        print("Please check the error messages above and try again.")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def post_training_analysis():
    """
    Perform post-training analysis to compare results
    """
    print()
    print("=== Post-Training Analysis ===")
    
    # Check if training was successful
    results_dir = Path("optimized_train/run1")
    if not results_dir.exists():
        print("Results directory not found. Training may not have completed.")
        return
    
    # Check for best model
    best_model = results_dir / "weights" / "best.pt"
    if best_model.exists():
        print(f"✅ Best model found: {best_model}")
    
    # Check for results
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        print(f"✅ Training results: {results_csv}")
    
    # Run validation on best model
    print()
    validation_command = [
        "yolo", "val",
        f"model={best_model}",
        "data=LATERAL ORTHO AI.v2i.yolov11/data.yaml",
        "imgsz=800",
        "device=0"
    ]
    
    print("Running validation on best model...")
    try:
        subprocess.run(validation_command, check=True, capture_output=False, text=True)
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    print("YOLOv11 Optimized Training Script")
    print("Based on comprehensive model analysis and recommendations")
    print("=" * 60)
    
    success = run_optimized_training()
    
    if success:
        post_training_analysis()
    
    print()
    print("Script completed. Check the model_analysis_report.md for detailed analysis.")
