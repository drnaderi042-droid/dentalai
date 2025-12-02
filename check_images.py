"""
Check required image files for TeethDreamer segmentation
"""
import os
from pathlib import Path

def check_images():
    image_dir = Path("my_images")
    required_files = ["0.png", "1.png", "2.png", "3.png", "4.png"]
    
    print("=" * 50)
    print("Checking Required Image Files")
    print("=" * 50)
    print()
    
    # Check if directory exists
    if not image_dir.exists():
        print(f"ERROR: Directory '{image_dir}' does not exist!")
        print(f"Full path: {image_dir.resolve()}")
        return False
    
    print(f"OK: Directory exists: {image_dir}")
    print(f"Full path: {image_dir.resolve()}")
    print()
    
    # Check required files
    print("Checking required files:")
    print()
    
    all_exist = True
    for file in required_files:
        file_path = image_dir / file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  OK: {file} - exists ({size_kb:.2f} KB)")
        else:
            print(f"  MISSING: {file} - not found!")
            all_exist = False
    
    print()
    
    # List all files
    print("All files in directory:")
    all_files = list(image_dir.glob("*"))
    if not all_files:
        print("  WARNING: Directory is empty!")
    else:
        for file in sorted(all_files):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name} ({size_kb:.2f} KB)")
    
    print()
    
    # Final result
    if all_exist:
        print("=" * 50)
        print("SUCCESS: All required files exist!")
        print("=" * 50)
        print()
        print("You can run segmentation:")
        print("  cd TeethDreamer")
        print("  python seg_teeth.py --img ../my_images --seg ../output/segmented --suffix png")
        return True
    else:
        print("=" * 50)
        print("ERROR: Some files are missing!")
        print("=" * 50)
        print()
        print(f"Please add the following files to '{image_dir}':")
        for file in required_files:
            if not (image_dir / file).exists():
                print(f"  - {file}")
        return False

if __name__ == "__main__":
    check_images()













