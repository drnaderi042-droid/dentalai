#!/usr/bin/env python3
"""
Check which exact model file is being loaded by unified_ai_api_server.py
and verify its parameters match the expected values (1024, 256)
"""

import os
import sys
import torch
from pathlib import Path

def main():
    print("=" * 80)
    print("EXACT MODEL LOADING DIAGNOSTIC")
    print("=" * 80)
    
    # Replicate the exact logic from unified_ai_api_server.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    # Check EXACT same candidates as unified_ai_api_server.py lines 704-707
    model_path_candidates = [
        os.path.join(base_dir, 'Aariz', 'models', 'hrnet_p1p2_heatmap_best.pth'),
        os.path.join(base_dir, 'aariz', 'models', 'hrnet_p1p2_heatmap_best.pth'),
    ]
    
    print(f"\n🔍 Checking EXACT candidates from unified_ai_api_server.py:")
    for i, candidate in enumerate(model_path_candidates, 1):
        exists = "✅ EXISTS" if os.path.exists(candidate) else "❌ NOT FOUND"
        print(f"   {i}. {exists}: {candidate}")
    
    # Try to find the first existing model
    model_path = None
    for candidate in model_path_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if not model_path:
        print(f"\n❌ ERROR: No P1/P2 model found in expected locations!")
        print(f"   This explains >100px errors - model cannot be loaded!")
        return
    
    print(f"\n📍 Using model path: {model_path}")
    print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Load checkpoint exactly as unified_ai_api_server.py does (line 730)
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully")
        
        # Get parameters exactly as unified_ai_api_server.py does (lines 732-733)
        p1p2_image_size = checkpoint.get('image_size', 1024)
        p1p2_heatmap_size = checkpoint.get('heatmap_size', 256)
        
        print(f"\n📊 MODEL PARAMETERS (from checkpoint):")
        print(f"   Image size: {p1p2_image_size}")
        print(f"   Heatmap size: {p1p2_heatmap_size}")
        
        # Check if parameters are CORRECT
        if p1p2_image_size == 1024 and p1p2_heatmap_size == 256:
            print(f"   ✅ Parameters are CORRECT (1024, 256)")
        else:
            print(f"   ❌ Parameters are WRONG! Expected (1024, 256)")
            print(f"   ❌ This will cause >100px coordinate errors!")
        
        # Check if there are other wrong models in the models directory
        models_dir = Path(model_path).parent
        print(f"\n🔍 Checking ALL P1/P2 models in {models_dir}:")
        
        for model_file in models_dir.glob('*p1p2*.pth'):
            try:
                ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
                img_size = ckpt.get('image_size', 'unknown')
                heat_size = ckpt.get('heatmap_size', 'unknown')
                
                status = "✅ CORRECT" if (img_size == 1024 and heat_size == 256) else "❌ WRONG"
                if img_size != 'unknown':
                    print(f"   {status} {model_file.name}: {img_size}x{heat_size}")
                else:
                    print(f"   ❓ UNKNOWN {model_file.name}: no parameters found")
            except Exception as e:
                print(f"   ❓ ERROR {model_file.name}: {str(e)}")
        
        # Final assessment
        print(f"\n{'='*80}")
        print(f"DIAGNOSTIC RESULT:")
        print(f"{'='*80}")
        
        if p1p2_image_size == 1024 and p1p2_heatmap_size == 256:
            print(f"✅ Model parameters are CORRECT (1024, 256)")
            print(f"✅ Model should produce ~2px accuracy")
            print(f"")
            print(f"   If >100px errors persist:")
            print(f"   1. Check if unified_ai_api_server.py is actually loading this model")
            print(f"   2. Check if there are multiple server instances running")
            print(f"   3. Check if web interface is calling different endpoints")
            print(f"   4. Check if coordinate transformation is being bypassed")
        else:
            print(f"❌ MODEL PARAMETERS ARE WRONG!")
            print(f"   Current: {p1p2_image_size}x{p1p2_heatmap_size}")
            print(f"   Expected: 1024x256")
            print(f"   ")
            print(f"❌ This EXPLAINs the >100px coordinate errors!")
            print(f"")
            print(f"🔧 SOLUTION:")
            print(f"   Update unified_ai_api_server.py to use the CORRECT model:")
            print(f"   'Aariz/models/hrnet_p1p2_heatmap_best.pth'")
        
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        print(f"   This explains >100px errors - model cannot be loaded!")

if __name__ == "__main__":
    main()