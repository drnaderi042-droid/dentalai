"""
Check all P1/P2 model files to see if any have wrong parameters
"""
import os
import torch

def check_model_file(filepath):
    """Check parameters in a model file"""
    print(f"\n{'='*50}")
    print(f"Checking: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Get basic info
        print(f"Keys: {list(checkpoint.keys())}")
        
        # Get parameters
        image_size = checkpoint.get('image_size', 'NOT FOUND')
        heatmap_size = checkpoint.get('heatmap_size', 'NOT FOUND')
        pixel_error = checkpoint.get('pixel_error', 'NOT FOUND')
        
        print(f"Image size: {image_size}")
        print(f"Heatmap size: {heatmap_size}")
        print(f"Pixel error: {pixel_error}")
        
        # Check if this matches the correct parameters
        if image_size == 1024 and heatmap_size == 256:
            print("✅ CORRECT parameters (1024, 256)")
            return True
        elif image_size == 768 and heatmap_size == 192:
            print("❌ WRONG parameters (768, 192) - This would cause >100px errors!")
            return False
        else:
            print(f"❓ UNKNOWN parameters ({image_size}, {heatmap_size})")
            return None
            
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return None

def main():
    models_dir = "Aariz/models"
    p1p2_files = [
        "hrnet_p1p2_heatmap_best.pth",
        "hrnet_p1p2_heatmap_best_optimized.pth", 
        "hrnet_p1p2_heatmap_best_quantized.pth",
        "p1p2_lightweight_mobilenet.pth",
        "p1p2_pruned.pth",
        "p1p2_quantized.pth",
        "p1p2_ultra_lightweight.pth"
    ]
    
    print("P1/P2 Model Parameter Analysis")
    print("Looking for models with wrong parameters that could cause >100px errors...")
    
    correct_models = []
    wrong_models = []
    unknown_models = []
    error_models = []
    
    for filename in p1p2_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            result = check_model_file(filepath)
            if result is True:
                correct_models.append(filename)
            elif result is False:
                wrong_models.append(filename)
            elif result is None:
                unknown_models.append(filename)
        else:
            print(f"[SKIP] {filename} not found")
            error_models.append(filename)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Correct models (1024, 256): {len(correct_models)}")
    for f in correct_models:
        print(f"  ✅ {f}")
    
    print(f"\nWrong models (768, 192): {len(wrong_models)}")
    for f in wrong_models:
        print(f"  ❌ {f}")
    
    print(f"\nUnknown models: {len(unknown_models)}")
    for f in unknown_models:
        print(f"  ❓ {f}")
    
    if wrong_models:
        print(f"\n🚨 FOUND THE ISSUE!")
        print(f"   Models with wrong parameters found: {wrong_models}")
        print(f"   These models would cause >100px coordinate errors!")
        print(f"   Check if unified_ai_api_server.py is loading one of these instead of the correct one.")
    else:
        print(f"\n✅ All found models have correct parameters")
        print(f"   If >100px errors persist, check:")
        print(f"   1. Which exact model file is being loaded by unified_ai_api_server.py")
        print(f"   2. Whether the server is using cached/stale model files")
        print(f"   3. Whether there are multiple server instances running")

if __name__ == "__main__":
    main()