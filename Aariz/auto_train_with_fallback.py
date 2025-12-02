"""
Automated training with intelligent fallback
Tests fixes first, then proceeds with full training or tries alternative approaches
"""
import sys
from pathlib import Path
import os
import torch
import time

sys.path.append(str(Path(__file__).parent))

from train_p1_p2_hrnet import train_hrnet_p1_p2_model

def test_training_20_epochs():
    """Test training for 20 epochs to verify fixes"""
    print("\n" + "="*60)
    print("PHASE 1: Testing Fixes (20 epochs)")
    print("="*60)
    
    model = train_hrnet_p1_p2_model(
        annotations_file='annotations_p1_p2.json',
        images_dir='Aariz/train/Cephalograms',
        output_dir='models',
        hrnet_variant='hrnet_w18',
        image_size=768,
        batch_size=4,
        num_epochs=20,
        learning_rate=0.005,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load best model to check metrics
    checkpoint_path = Path('models/hrnet_p1p2_best_hrnet_w18.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pixel_error = checkpoint.get('pixel_error', 999)
        val_loss = checkpoint.get('val_loss', 999)
        
        print("\n" + "="*60)
        print("TEST RESULTS:")
        print("="*60)
        print(f"  Pixel Error: {pixel_error:.2f} px")
        print(f"  Val Loss: {val_loss:.6f}")
        
        return pixel_error, val_loss
    
    return 999, 999

def full_training():
    """Full training with 200 epochs"""
    print("\n" + "="*60)
    print("PHASE 2: Full Training (200 epochs)")
    print("="*60)
    
    model = train_hrnet_p1_p2_model(
        annotations_file='annotations_p1_p2.json',
        images_dir='Aariz/train/Cephalograms',
        output_dir='models',
        hrnet_variant='hrnet_w18',
        image_size=768,
        batch_size=2,  # Reduced for stability
        num_epochs=200,
        learning_rate=0.003,  # Slightly lower for fine-tuning
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return model

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    start_time = time.time()
    
    # Check files
    if not Path('annotations_p1_p2.json').exists():
        print("ERROR: annotations_p1_p2.json not found!")
        sys.exit(1)
    
    if not Path('Aariz/train/Cephalograms').exists():
        print("ERROR: Images directory not found!")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" "*15 + "AUTOMATED TRAINING WITH FALLBACK")
    print("="*70)
    print("\nStrategy:")
    print("  1. Test for 20 epochs (~15 min)")
    print("  2. If pixel error < 50px: Continue with 200 epochs")
    print("  3. If pixel error > 50px: Report issue (need Plan B)")
    print("\nFixes Applied:")
    print("  ✓ Augmentation disabled (was causing landmark mismatch)")
    print("  ✓ Pixel error calculation fixed")
    print("  ✓ Learning rate optimized")
    print("="*70)
    
    # Phase 1: Test
    pixel_error, val_loss = test_training_20_epochs()
    
    test_time = (time.time() - start_time) / 60
    print(f"\nTest phase completed in {test_time:.1f} minutes")
    
    # Decision point
    if pixel_error < 50:
        print("\n" + "="*70)
        print("✓ TEST PASSED! Pixel error is acceptable.")
        print("="*70)
        print(f"\nResults after 20 epochs:")
        print(f"  - Pixel Error: {pixel_error:.2f} px  ← Good!")
        print(f"  - Val Loss: {val_loss:.6f}")
        print("\nProceeding with full training (200 epochs)...")
        print("ETA: ~2-3 hours")
        print("="*70)
        
        # Backup test model
        test_model_path = Path('models/hrnet_p1p2_test_20epochs.pth')
        final_model_path = Path('models/hrnet_p1p2_best_hrnet_w18.pth')
        if final_model_path.exists():
            import shutil
            shutil.copy(final_model_path, test_model_path)
            print(f"\nTest model backed up to: {test_model_path}")
        
        # Phase 2: Full training
        time.sleep(5)  # Brief pause
        model = full_training()
        
        total_time = (time.time() - start_time) / 60
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nTotal time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
        print(f"\nModel saved to: models/hrnet_p1p2_best_hrnet_w18.pth")
        print("\nNext steps:")
        print("  1. Test model: python test_p1_p2_hrnet.py")
        print("  2. Check results: test_results_hrnet/")
        print("  3. Integrate into frontend")
        print("="*70)
        
    else:
        print("\n" + "="*70)
        print("⚠ TEST INCONCLUSIVE - Need different approach")
        print("="*70)
        print(f"\nResults after 20 epochs:")
        print(f"  - Pixel Error: {pixel_error:.2f} px  ← Still high!")
        print(f"  - Val Loss: {val_loss:.6f}")
        print("\n" + "="*70)
        print("ANALYSIS:")
        print("="*70)
        
        if pixel_error > 150:
            print("\n❌ Direct regression approach may not be optimal")
            print("\nRecommendations:")
            print("  1. Check annotation quality:")
            print("     python check_annotations_quality.py annotations_p1_p2.json")
            print()
            print("  2. Try heatmap-based approach (more robust)")
            print("     → Implement heatmap regression instead of coordinate regression")
            print()
            print("  3. Verify dataset:")
            print("     - Are p1/p2 consistently labeled?")
            print("     - Are coordinates in correct format?")
            print("     - Check a few samples manually")
            print()
            print("  4. Consider two-stage detection:")
            print("     - Stage 1: Detect ruler region")
            print("     - Stage 2: Detect p1/p2 in region")
        elif pixel_error > 75:
            print("\n⚠ Moderate improvement but not sufficient")
            print("\nSuggestions:")
            print("  1. More training data (current: 100 images)")
            print("     → Annotate 200+ images for better results")
            print()
            print("  2. Try different architecture:")
            print("     → HRNet-W32 or ResNet with better feature extraction")
            print()
            print("  3. Adjust loss function:")
            print("     → Try Smooth L1 Loss or Wing Loss")
        else:
            print("\n🤔 Close to threshold - might work with more epochs")
            print("\nOptions:")
            print("  1. Continue training for 50 more epochs")
            print("  2. Fine-tune hyperparameters")
            print("  3. Add more data augmentation (carefully!)")
        
        print("\n" + "="*70)
        print("Current model saved for analysis")
        print("Review results and decide next steps")
        print("="*70)
    
    print("\n" + "="*70)
    print("Script execution completed")
    print(f"Check AUTOMATED_FIX_REPORT.md for details")
    print("="*70)













