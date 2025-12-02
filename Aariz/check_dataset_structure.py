"""
Check if the Aariz dataset structure is correct
"""

import os
from pathlib import Path

def check_dataset_structure():
    """Check if all required directories exist."""
    print("🔍 Checking Aariz dataset structure...\n")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📂 Current directory: {os.getcwd()}\n")
    
    # Check required paths
    checks = [
        ("Aariz dataset root", Path("Aariz")),
        ("Train directory", Path("Aariz/train")),
        ("Cephalograms directory", Path("Aariz/train/Cephalograms")),
        ("Annotations directory", Path("Aariz/train/Annotations")),
        ("Cephalometric Landmarks", Path("Aariz/train/Annotations/Cephalometric Landmarks")),
        ("Senior Orthodontists", Path("Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists")),
    ]
    
    all_good = True
    for name, path in checks:
        if path.exists():
            if path.is_dir():
                # Count files
                files = list(path.glob("*"))
                print(f"✅ {name}: {path}")
                print(f"   └─ {len(files)} items inside\n")
            else:
                print(f"⚠️ {name}: exists but is not a directory\n")
                all_good = False
        else:
            print(f"❌ {name}: NOT FOUND")
            print(f"   Expected at: {path.absolute()}\n")
            all_good = False
    
    if all_good:
        # Check for p1/p2 annotations
        annotations_dir = Path("Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists")
        
        p1p2_image_ids = [
            'cks2ip8fq29yq0yufc4scftj8',
            'cks2ip8fq29z00yufgnfla2tf',
            'cks2ip8fq29za0yuf0tqu1qjs',
            'cks2ip8fq2a0j0yufdfssbc09',
            'cks2ip8fq2a0t0yufgab484s9',
            'cks2ip8fq2a130yuf5gyh2nrs',
            'cks2ip8fq2a180yufh98ue4yo',
            'cks2ip8fq2a1i0yuf9ra939xh',
            'cks2ip8fq2a1n0yuf8nqt3ndt',
            'cks2ip8fq2a1x0yuffrma5nom',
            'cks2ip8fr2a2c0yuf3pc66vjh',
            'cks2ip8fr2a2h0yuf2r8o8teg',
            'cks2ip8fr2a2m0yuf7tz6ci2u',
            'cks2ip8fr2a2w0yuf49bu0v1w',
            'cks2ip8fr2a3b0yuff9a6ac73',
            'cks2ip8fr2a3l0yuf8pbcfolv',
            'cks2ip8fr2a3q0yufadyu84rc',
            'cks2ip8fr2a3v0yuf4hws1b5t',
        ]
        
        print(f"📊 Checking for P1/P2 annotations...\n")
        found_count = 0
        missing = []
        
        for image_id in p1p2_image_ids:
            annotation_file = annotations_dir / f"{image_id}.json"
            if annotation_file.exists():
                found_count += 1
            else:
                missing.append(image_id)
        
        print(f"✅ Found {found_count}/{len(p1p2_image_ids)} P1/P2 annotation files\n")
        
        if missing:
            print(f"⚠️ Missing annotations for:")
            for img_id in missing:
                print(f"   - {img_id}")
            print()
        
        # Check for corresponding images
        images_dir = Path("Aariz/train/Cephalograms")
        print(f"📷 Checking for corresponding images...\n")
        
        found_images = 0
        for image_id in p1p2_image_ids:
            found_img = False
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = images_dir / f"{image_id}{ext}"
                if img_path.exists():
                    found_images += 1
                    found_img = True
                    break
            
            if not found_img:
                print(f"⚠️ Image not found for: {image_id}")
        
        print(f"✅ Found {found_images}/{len(p1p2_image_ids)} images\n")
        
        if found_count > 0 and found_images > 0:
            print("=" * 60)
            print("✅ Dataset structure is CORRECT!")
            print("=" * 60)
            print(f"\nYou can now run:")
            print(f"  python test_calibration_detection.py")
            print(f"or")
            print(f"  python quick_test_calibration.py")
        else:
            print("=" * 60)
            print("⚠️ Dataset is incomplete")
            print("=" * 60)
    else:
        print("=" * 60)
        print("❌ Dataset structure is INCORRECT")
        print("=" * 60)
        print(f"\nPlease check that you have the Aariz dataset in the correct location.")
        print(f"Expected structure:")
        print(f"  aariz/")
        print(f"  └── Aariz/")
        print(f"      └── train/")
        print(f"          ├── Cephalograms/")
        print(f"          └── Annotations/")
        print(f"              └── Cephalometric Landmarks/")
        print(f"                  └── Senior Orthodontists/")

if __name__ == '__main__':
    check_dataset_structure()

