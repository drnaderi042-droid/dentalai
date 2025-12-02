"""
Convert SAM model from directory structure to .pth file
"""
import torch
import os
import sys

def convert_sam_model():
    # مسیرها
    source_dir = "3D/sam_vit_b_01ec64/archive"
    target_file = "TeethDreamer/ckpt/sam_vit_b_01ec64.pth"
    
    # بررسی وجود source
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found!")
        return False
    
    # ایجاد پوشه target
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    try:
        print(f"Loading model from: {source_dir}")
        # بارگذاری از directory (PyTorch archive format)
        model = torch.load(source_dir, map_location="cpu")
        
        print(f"Saving to: {target_file}")
        # ذخیره به صورت .pth
        torch.save(model, target_file)
        
        # بررسی وجود فایل
        if os.path.exists(target_file):
            size_mb = os.path.getsize(target_file) / (1024 * 1024)
            print(f"Conversion completed successfully!")
            print(f"File size: {size_mb:.2f} MB")
            return True
        else:
            print("Error: File was not created!")
            return False
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nAlternative: Download the .pth file directly:")
        print("   URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"   Save to: {target_file}")
        return False

if __name__ == "__main__":
    success = convert_sam_model()
    sys.exit(0 if success else 1)

