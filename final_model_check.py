#!/usr/bin/env python3
"""
Check parameters of ALL P1/P2 models to identify the wrong ones causing >100px errors
"""

import os
import sys
import torch
from pathlib import Path

def check_model_parameters(model_path):
    """Check parameters of a single model"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        image_size = checkpoint.get('image_size', 'unknown')
        heatmap_size = checkpoint.get('heatmap_size', 'unknown')
        return image_size, heatmap_size
    except Exception as e:
        return 'error', str(e)

def main():
    print("=" * 80)
    print("P1/P2 MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    
    models_dir = Path("Aariz/models")
    p1p2_models = [f for f in models_dir.glob("*p1p2*.pth")]
    
    correct_models = []
    wrong_models = []
    error_models = []
    
    for model_file in sorted(p1p2_models):
        print(f"\nChecking: {model_file.name}")
        img_size, heat_size = check_model_parameters(model_file)
        
        if img_size == 'error':
            print(f"  ERROR: {heat_size}")
            error_models.append(model_file.name)
        elif img_size == 1024 and heat_size == 256:
            print(f"  CORRECT: {img_size}x{heat_size}")
            correct_models.append(model_file.name)
        else:
            print(f"  WRONG: {img_size}x{heat_size} (Expected: 1024x256)")
            wrong_models.append(model_file.name)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nCORRECT Models ({len(correct_models)}):")
    for model in correct_models:
        print(f"  - {model}")
    
    print(f"\nWRONG Models ({len(wrong_models)}):")
    for model in wrong_models:
        print(f"  - {model} (These cause >100px errors!)")
    
    if error_models:
        print(f"\nERROR Models ({len(error_models)}):")
        for model in error_models:
            print(f"  - {model}")
    
    print(f"\n{'='*80}")
    print(f"ROOT CAUSE ANALYSIS")
    print(f"{'='*80}")
    
    if wrong_models:
        print(f"FOUND THE PROBLEM!")
        print(f"The following models have WRONG parameters (not 1024x256):")
        for model in wrong_models:
            print(f"  - {model}")
        
        print(f"\nThese models cause >100px coordinate errors because:")
        print(f"  - Training: Coordinates normalized to 1024x1024 image")
        print(f"  - Inference: Image resized to wrong size (768 instead of 1024)")
        print(f"  - Result: Coordinates are scaled incorrectly")
        
        print(f"\nSUSPECTED CAUSE:")
        print(f"  The unified_ai_api_server.py might be loading one of these WRONG models")
        print(f"  instead of the correct 'hrnet_p1p2_heatmap_best.pth'")
        
        print(f"\nSOLUTION:")
        print(f"  1. Ensure unified_ai_api_server.py uses EXACTLY: 'Aariz/models/hrnet_p1p2_heatmap_best.pth'")
        print(f"  2. Delete or rename the WRONG models to avoid confusion:")
        for model in wrong_models:
            print(f"     - Rename '{model}' to '{model}.WRONG'")
    else:
        print(f"ALL P1/P2 MODELS HAVE CORRECT PARAMETERS (1024x256)")
        print(f"")
        print(f"If >100px errors persist, the issue is NOT model parameters.")
        print(f"Possible causes:")
        print(f"  1. Wrong model file is being loaded by unified_ai_api_server.py")
        print(f"  2. Multiple server instances running with different models")
        print(f"  3. Web interface calling wrong API endpoint")
        print(f"  4. Cached/stale model data being used")

if __name__ == "__main__":
    main()