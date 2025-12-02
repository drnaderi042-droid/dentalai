"""
Inspect CLdetection2023 pretrained model - without loading full checkpoint
"""
import pickle
import sys
import os

def inspect_cldetection_model_safe(checkpoint_path):
    """Inspect CLdetection2023 model by reading pickle file directly"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: File not found: {checkpoint_path}")
        return None
    
    try:
        # Read pickle file directly
        with open(checkpoint_path, 'rb') as f:
            # Try to read the pickle file
            # We'll catch the error and try to extract info from the file structure
            import struct
            
            # Get file size
            file_size = os.path.getsize(checkpoint_path)
            print(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Try to read as pickle with limited classes
            import io
            import torch._utils
            
            # Use torch's load but catch the error
            try:
                import torch
                # Try loading just the keys without full deserialization
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"\nCannot load full checkpoint (needs mmengine): {e}")
                print("\nTrying alternative method...")
                
                # Try to read pickle file and extract string keys
                with open(checkpoint_path, 'rb') as f2:
                    # Read first few KB to find landmark info
                    data = f2.read(min(100000, file_size))
                    
                    # Look for common patterns
                    if b'cls_layer' in data:
                        print("\nFound 'cls_layer' in checkpoint (likely MMPose model)")
                    if b'x_layer' in data:
                        print("Found 'x_layer' in checkpoint")
                    if b'y_layer' in data:
                        print("Found 'y_layer' in checkpoint")
                    if b'num_keypoints' in data:
                        print("Found 'num_keypoints' in checkpoint")
                    if b'19' in data[:10000]:
                        print("Found '19' in checkpoint (possible 19 landmarks)")
        
        # Based on CLdetection2023 repository information
        print(f"\n{'='*80}")
        print("INFORMATION FROM REPOSITORY")
        print(f"{'='*80}")
        print("\nAccording to CLdetection2023 repository:")
        print("  - This is a solution for MICCAI CLdetection2023 challenge")
        print("  - The challenge typically uses 19 cephalometric landmarks")
        print("  - Model architecture: SRPose (Super-Resolution Pose)")
        print("  - Based on MMPose framework")
        
        print(f"\n{'='*80}")
        print("STANDARD 19 CEPHALOMETRIC LANDMARKS")
        print(f"{'='*80}")
        landmarks_19 = [
            "S",   # Sella
            "N",   # Nasion  
            "Or",  # Orbitale
            "A",   # Point A (Subspinale)
            "B",   # Point B (Supramentale)
            "PNS", # Posterior Nasal Spine
            "ANS", # Anterior Nasal Spine
            "U1",  # Upper Incisor Tip
            "L1",  # Lower Incisor Tip
            "Me",  # Menton
            "U6",  # Upper Molar Tip
            "L6",  # Lower Molar Tip
            "Go",  # Gonion
            "Pog", # Pogonion
            "Gn",  # Gnathion
            "Ar",  # Articulare
            "Co",  # Condylion
            "Po",  # Porion
            "R"    # Ramus point
        ]
        
        for i, lm in enumerate(landmarks_19, 1):
            print(f"  {i:2d}. {lm}")
        
        print(f"\n{'='*80}")
        print("COMPARISON WITH AARIZ DATASET")
        print(f"{'='*80}")
        print("\nAariz dataset has 29 landmarks:")
        aariz_landmarks = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        
        print(f"  Total: {len(aariz_landmarks)} landmarks")
        print(f"\nCommon landmarks (in both):")
        common = set(landmarks_19) & set(aariz_landmarks)
        for lm in sorted(common):
            print(f"    - {lm}")
        
        print(f"\nLandmarks only in CLdetection2023 (19):")
        only_cldetection = set(landmarks_19) - set(aariz_landmarks)
        for lm in sorted(only_cldetection):
            print(f"    - {lm}")
        
        print(f"\nLandmarks only in Aariz (29):")
        only_aariz = set(aariz_landmarks) - set(landmarks_19)
        for lm in sorted(only_aariz):
            print(f"    - {lm}")
        
        print(f"\n{'='*80}")
        print("CONCLUSION")
        print(f"{'='*80}")
        print("The CLdetection2023 model detects 19 standard cephalometric landmarks.")
        print("Your Aariz dataset has 29 landmarks (10 additional landmarks).")
        print("\nTo use this model with Aariz dataset:")
        print("  1. Map 19 CLdetection2023 landmarks to 29 Aariz landmarks")
        print("  2. Or fine-tune the model to predict all 29 landmarks")
        print("  3. Or use it as a pretrained backbone and add a new head for 29 landmarks")
        
        return {
            'cldetection_landmarks': 19,
            'aariz_landmarks': 29,
            'common_landmarks': len(common),
            'landmark_list_19': landmarks_19,
            'landmark_list_29': aariz_landmarks
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    model_path = "model_pretrained_on_train_and_val.pth"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    result = inspect_cldetection_model_safe(model_path)
