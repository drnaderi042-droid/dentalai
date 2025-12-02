"""
Test CLdetection2023 model accuracy on Aariz dataset
Uses the original CLdetection2023 repository for inference
Then compares with Aariz ground truth for common landmarks
"""
import os
import sys
import json
import numpy as np
import glob
from tqdm import tqdm

# Common landmarks between CLdetection2023 (19) and Aariz (29)
COMMON_LANDMARKS = [
    "S", "N", "Or", "A", "B", "PNS", "ANS", "Me", 
    "Go", "Pog", "Gn", "Ar", "Co", "Po", "R"
]

# Mapping: CLdetection2023 index -> Aariz landmark name
CLDETECTION_TO_AARIZ_MAPPING = {
    0: "S",    # S
    1: "N",    # N
    2: "Or",   # Or
    3: "A",    # A
    4: "B",    # B
    5: "PNS",  # PNS
    6: "ANS",  # ANS
    9: "Me",   # Me (skip U1=7, L1=8)
    12: "Go",  # Go (skip U6=10, L6=11)
    13: "Pog", # Pog
    14: "Gn",  # Gn
    15: "Ar",  # Ar
    16: "Co",  # Co
    17: "Po",  # Po
    18: "R"    # R
}


def load_ground_truth_aariz(image_id, dataset_path="Aariz", annotation_type="Senior Orthodontists"):
    """Load ground truth landmarks from Aariz dataset"""
    for mode in ["test", "valid", "train"]:
        gt_path = os.path.join(
            dataset_path, mode, "Annotations", "Cephalometric Landmarks",
            annotation_type, f"{image_id}.json"
        )
        if os.path.exists(gt_path):
            break
    
    if not os.path.exists(gt_path):
        return None, None, None
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # Get pixel size
    pixel_size = 0.1  # default
    csv_path = os.path.join(dataset_path, "cephalogram_machine_mappings.csv")
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            row = df[df['cephalogram_id'] == image_id]
            if len(row) > 0:
                pixel_size = row.iloc[0]['pixel_size']
        except:
            pass
    
    # Get original image size
    from PIL import Image
    for mode in ["test", "valid", "train"]:
        img_path = os.path.join(dataset_path, mode, "Cephalograms", f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_path, mode, "Cephalograms", f"{image_id}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            orig_size = img.size  # (width, height)
            break
    
    landmarks_dict = {}
    for lm in annotation.get('landmarks', []):
        symbol = lm['symbol']
        landmarks_dict[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return landmarks_dict, pixel_size, orig_size


def get_test_images_aariz(dataset_path="Aariz", num_images=10):
    """Get test images from Aariz dataset"""
    test_images = []
    
    for mode in ["test", "valid", "train"]:
        images_folder = os.path.join(dataset_path, mode, "Cephalograms")
        if not os.path.exists(images_folder):
            continue
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
        for img_path in image_files[:num_images * 2]:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            gt, pixel_size, orig_size = load_ground_truth_aariz(img_id, dataset_path)
            if gt:
                test_images.append({
                    'image_path': img_path,
                    'image_id': img_id,
                    'gt_landmarks': gt,
                    'pixel_size': pixel_size,
                    'orig_size': orig_size
                })
                if len(test_images) >= num_images:
                    break
        
        if len(test_images) >= num_images:
            break
    
    return test_images


def run_cldetection_inference(image_path, cldetection_repo_path=None):
    """
    Run CLdetection2023 inference on a single image
    Returns predictions in format: {landmark_name: {'x': float, 'y': float}}
    """
    # Try to find CLdetection2023 repository
    if cldetection_repo_path is None:
        possible_paths = [
            "../CLdetection2023",
            "../../CLdetection2023",
            "CLdetection2023",
            os.path.join(os.path.dirname(__file__), "..", "CLdetection2023")
        ]
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "inference_single_image.py")):
                cldetection_repo_path = path
                break
    
    if cldetection_repo_path is None or not os.path.exists(cldetection_repo_path):
        return None
    
    # Try to use MMPose API
    try:
        import sys
        sys.path.insert(0, cldetection_repo_path)
        
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules
        register_all_modules()
        
        # Initialize model
        config_path = os.path.join(cldetection_repo_path, "configs", "CLdetection2023", "srpose_s2.py")
        model_path = os.path.join(os.path.dirname(__file__), "model_pretrained_on_train_and_val.pth")
        
        if not os.path.exists(config_path):
            return None
        
        if not os.path.exists(model_path):
            # Try relative to repo
            model_path = os.path.join(cldetection_repo_path, "model_pretrained_on_train_and_val.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), "model_pretrained_on_train_and_val.pth")
        
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = init_model(config_path, model_path, device=device)
        
        # Run inference
        results = inference_topdown(model, image_path)
        
        # Extract keypoints
        if isinstance(results, list) and len(results) > 0:
            pred_keypoints = results[0].get('pred_instances', {}).get('keypoints', None)
            if pred_keypoints is not None:
                pred_keypoints = pred_keypoints[0].cpu().numpy()  # Shape: [19, 2] or [19, 3]
                if len(pred_keypoints.shape) == 2:
                    if pred_keypoints.shape[1] == 3:
                        pred_keypoints = pred_keypoints[:, :2]  # Remove visibility
                    
                    # Map to Aariz landmarks
                    pred_mapped = {}
                    for cld_idx, aariz_lm in CLDETECTION_TO_AARIZ_MAPPING.items():
                        if cld_idx < len(pred_keypoints):
                            pred_mapped[aariz_lm] = {
                                'x': float(pred_keypoints[cld_idx][0]),
                                'y': float(pred_keypoints[cld_idx][1])
                            }
                    
                    return pred_mapped
        
        return None
        
    except Exception as e:
        print(f"Error running inference: {e}")
        return None


def test_cldetection_accuracy_manual(aariz_path="Aariz", num_images=10):
    """
    Manual testing guide - since MMPose may not be available
    """
    print("="*80)
    print("CLdetection2023 Model Accuracy Test on Aariz Dataset")
    print("="*80)
    
    print("\n" + "="*80)
    print("SETUP INSTRUCTIONS")
    print("="*80)
    print("\nTo test the CLdetection2023 model, you need to:")
    print("\n1. Clone the CLdetection2023 repository:")
    print("   git clone https://github.com/5k5000/CLdetection2023.git")
    print("   cd CLdetection2023")
    
    print("\n2. Install dependencies:")
    print("   conda create -n LMD python=3.10")
    print("   conda activate LMD")
    print("   pip install -r requirements.txt")
    print("   pip install -U openmim")
    print("   cd mmpose_package/mmpose")
    print("   pip install -e .")
    print("   mim install mmengine")
    print("   mim install 'mmcv>=2.0.0'")
    
    print("\n3. Copy the model file:")
    print("   cp ../Aariz/model_pretrained_on_train_and_val.pth ./")
    
    print("\n4. Run inference on Aariz images:")
    print("   python inference_single_image.py \\")
    print("       --config 'configs/CLdetection2023/srpose_s2.py' \\")
    print("       --checkpoint 'model_pretrained_on_train_and_val.pth' \\")
    print("       --image_path '../Aariz/test/Cephalograms/image_id.png'")
    
    print("\n" + "="*80)
    print("AUTOMATED TEST SCRIPT")
    print("="*80)
    print("\nAlternatively, if MMPose is installed, run:")
    print("   python test_cldetection_accuracy.py --use_mmpose")
    
    # Get test images info
    print("\n" + "="*80)
    print("TEST IMAGES INFORMATION")
    print("="*80)
    test_images = get_test_images_aariz(aariz_path, num_images)
    print(f"\nFound {len(test_images)} test images with ground truth")
    
    if len(test_images) > 0:
        print("\nSample test images:")
        for i, item in enumerate(test_images[:5], 1):
            print(f"  {i}. {item['image_id']}")
            print(f"     Path: {item['image_path']}")
            print(f"     Pixel size: {item['pixel_size']} mm/pixel")
            print(f"     Original size: {item['orig_size']}")
            
            # Check which common landmarks are available
            available_common = [lm for lm in COMMON_LANDMARKS if lm in item['gt_landmarks']]
            print(f"     Available common landmarks: {len(available_common)}/15")
    
    print("\n" + "="*80)
    print("EXPECTED OUTPUT FORMAT")
    print("="*80)
    print("\nThe CLdetection2023 model outputs 19 landmarks:")
    cldetection_landmarks = [
        "S", "N", "Or", "A", "B", "PNS", "ANS", "U1", "L1", "Me",
        "U6", "L6", "Go", "Pog", "Gn", "Ar", "Co", "Po", "R"
    ]
    for i, lm in enumerate(cldetection_landmarks):
        marker = "[COMMON]" if lm in COMMON_LANDMARKS else "[NOT IN AARIZ]"
        print(f"  {i:2d}. {marker} {lm}")
    
    print("\nCommon landmarks (15) that will be evaluated:")
    for i, lm in enumerate(COMMON_LANDMARKS, 1):
        print(f"  {i:2d}. {lm}")
    
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print("\nThe evaluation will calculate:")
    print("  - Mean Radial Error (MRE) in millimeters")
    print("  - Median Error")
    print("  - Standard Deviation")
    print("  - Success Detection Rate (SDR) at thresholds: 1mm, 2mm, 2.5mm, 3mm, 4mm")
    print("  - Per-landmark statistics")
    
    print("\n" + "="*80)


def test_cldetection_accuracy_with_mmpose(model_path, aariz_path="Aariz", num_images=10, cldetection_repo_path=None):
    """
    Test CLdetection2023 model accuracy using MMPose
    """
    print("="*80)
    print("Testing CLdetection2023 Model Accuracy on Aariz Dataset")
    print("="*80)
    
    # Get test images
    print(f"\nLoading test images from {aariz_path}...")
    test_images = get_test_images_aariz(aariz_path, num_images)
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    # Evaluate
    all_errors_mm = []
    landmark_errors = {lm: [] for lm in COMMON_LANDMARKS}
    successful_predictions = 0
    
    print("\nRunning inference...")
    for item in tqdm(test_images):
        image_path = item['image_path']
        image_id = item['image_id']
        gt_landmarks = item['gt_landmarks']
        pixel_size = item['pixel_size']
        
        # Run inference
        pred_mapped = run_cldetection_inference(image_path, cldetection_repo_path)
        
        if pred_mapped is None:
            continue
        
        successful_predictions += 1
        
        # Calculate errors for common landmarks
        for lm_name in COMMON_LANDMARKS:
            if lm_name in pred_mapped and lm_name in gt_landmarks:
                pred = pred_mapped[lm_name]
                gt = gt_landmarks[lm_name]
                
                # Calculate error in pixels
                error_px = np.sqrt(
                    (pred['x'] - gt['x'])**2 + 
                    (pred['y'] - gt['y'])**2
                )
                
                # Convert to mm
                error_mm = error_px * pixel_size
                
                landmark_errors[lm_name].append(error_mm)
                all_errors_mm.append(error_mm)
    
    if len(all_errors_mm) == 0:
        print("\nNo valid predictions found!")
        print("Make sure MMPose is installed and CLdetection2023 repository is available.")
        return
    
    # Calculate statistics
    print("\n" + "="*80)
    print("RESULTS - Common Landmarks Only (15 landmarks)")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Successful predictions: {successful_predictions}/{len(test_images)}")
    print(f"  Total landmark predictions: {len(all_errors_mm)}")
    print(f"  Mean Error (MRE): {np.mean(all_errors_mm):.2f} mm")
    print(f"  Median Error: {np.median(all_errors_mm):.2f} mm")
    print(f"  Std Error: {np.std(all_errors_mm):.2f} mm")
    print(f"  Min Error: {np.min(all_errors_mm):.2f} mm")
    print(f"  Max Error: {np.max(all_errors_mm):.2f} mm")
    
    # SDR
    thresholds = [1.0, 2.0, 2.5, 3.0, 4.0]
    print(f"\nSuccess Detection Rate (SDR):")
    for threshold in thresholds:
        sdr = (np.array(all_errors_mm) <= threshold).sum() / len(all_errors_mm) * 100
        print(f"  SDR @ {threshold}mm: {sdr:.2f}%")
    
    # Per-landmark statistics
    print(f"\nPer-Landmark Statistics:")
    print(f"{'Landmark':<10} {'Count':<8} {'MRE (mm)':<12} {'Median (mm)':<12} {'SDR@2mm':<10}")
    print("-" * 60)
    
    for lm_name in COMMON_LANDMARKS:
        if len(landmark_errors[lm_name]) > 0:
            errors = np.array(landmark_errors[lm_name])
            mre = np.mean(errors)
            median = np.median(errors)
            sdr_2mm = (errors <= 2.0).sum() / len(errors) * 100
            print(f"{lm_name:<10} {len(errors):<8} {mre:<12.2f} {median:<12.2f} {sdr_2mm:<10.1f}")
        else:
            print(f"{lm_name:<10} {'0':<8} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test CLdetection2023 model on Aariz dataset')
    parser.add_argument('--model_path', type=str, default='model_pretrained_on_train_and_val.pth',
                       help='Path to CLdetection2023 model')
    parser.add_argument('--aariz_path', type=str, default='Aariz',
                       help='Path to Aariz dataset')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of test images')
    parser.add_argument('--use_mmpose', action='store_true',
                       help='Use MMPose for inference (requires installation)')
    parser.add_argument('--cldetection_repo', type=str, default=None,
                       help='Path to CLdetection2023 repository')
    
    args = parser.parse_args()
    
    if args.use_mmpose:
        test_cldetection_accuracy_with_mmpose(
            args.model_path, 
            args.aariz_path, 
            args.num_images,
            args.cldetection_repo
        )
    else:
        test_cldetection_accuracy_manual(args.aariz_path, args.num_images)
