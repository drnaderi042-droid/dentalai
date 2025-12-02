"""
Quick test training for 20 epochs to verify fixes
"""
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from train_p1_p2_hrnet import train_hrnet_p1_p2_model

if __name__ == '__main__':
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'Aariz/train/Cephalograms'
    
    # Check if files exist
    if not Path(annotations_file).exists():
        print(f"ERROR: Annotations file not found: {annotations_file}")
        sys.exit(1)
    
    if not Path(images_dir).exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    print("="*60)
    print("QUICK TEST: 20 Epochs Training")
    print("="*60)
    print("\nConfiguration:")
    print("  - Epochs: 20 (for quick test)")
    print("  - Image Size: 768px")
    print("  - Batch Size: 4 (faster)")
    print("  - Learning Rate: 0.005 (higher for faster convergence)")
    print("  - Augmentation: DISABLED (fixed mismatch issue)")
    print("\nExpected after 20 epochs:")
    print("  - Pixel Error should drop below 50px")
    print("  - If not, will try different approach")
    print("="*60)
    print()
    
    # Train model with test settings
    model = train_hrnet_p1_p2_model(
        annotations_file=annotations_file,
        images_dir=images_dir,
        output_dir='models',
        hrnet_variant='hrnet_w18',
        image_size=768,
        batch_size=4,  # Increased for faster training
        num_epochs=20,  # Only 20 epochs for testing
        learning_rate=0.005,  # Higher LR for faster convergence
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nCheck the final pixel error above.")
    print("If < 50px: Training is working! ✓")
    print("If > 100px: Need different approach...")
    print("="*60)













