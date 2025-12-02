"""
Test CLdetection2023 model on Aariz dataset using the original repository
This script uses the CLdetection2023 inference script and compares with ground truth
"""
import os
import sys
import json
import numpy as np
import glob
from tqdm import tqdm

# Add CLdetection2023 to path
CLDETECTION_REPO = os.path.join(os.path.dirname(__file__), "..", "CLdetection2023")
if os.path.exists(CLDETECTION_REPO):
    sys.path.insert(0, CLDETECTION_REPO)
    sys.path.insert(0, os.path.join(CLDETECTION_REPO, "mmpose_package", "mmpose"))

# Common landmarks mapping
COMMON_LANDMARKS = [
    "S", "N", "Or", "A", "B", "PNS", "ANS", "Me", 
    "Go", "Pog", "Gn", "Ar", "Co", "Po", "R"
]

CLDETECTION_TO_AARIZ_MAPPING = {
    0: "S", 1: "N", 2: "Or", 3: "A", 4: "B", 5: "PNS", 6: "ANS",
    9: "Me", 12: "Go", 13: "Pog", 14: "Gn", 15: "Ar", 16: "Co", 17: "Po", 18: "R"
}

CLDETECTION_19_LANDMARKS = [
    "S", "N", "Or", "A", "B", "PNS", "ANS", "U1", "L1", "Me",
    "U6", "L6", "Go", "Pog", "Gn", "Ar", "Co", "Po", "R"
]


def load_ground_truth_aariz(image_id, dataset_path="Aariz"):
    """Load ground truth from Aariz dataset"""
    for mode in ["test", "valid", "train"]:
        gt_path = os.path.join(
            dataset_path, mode, "Annotations", "Cephalometric Landmarks",
            "Senior Orthodontists", f"{image_id}.json"
        )
        if os.path.exists(gt_path):
            break
    
    if not os.path.exists(gt_path):
        return None, None, None
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # Get pixel size
    pixel_size = 0.1
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
    
    # Get original size
    from PIL import Image
    for mode in ["test", "valid", "train"]:
        img_path = os.path.join(dataset_path, mode, "Cephalograms", f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_path, mode, "Cephalograms", f"{image_id}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            orig_size = img.size
            break
    
    landmarks_dict = {}
    for lm in annotation.get('landmarks', []):
        symbol = lm['symbol']
        landmarks_dict[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return landmarks_dict, pixel_size, orig_size


def get_test_images(dataset_path="Aariz", num_images=10):
    """Get test images"""
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


def run_inference_using_repo(image_path, config_path, model_path, cldetection_repo):
    """Run inference using CLdetection2023 repository"""
    try:
        from mmengine.config import Config
        from mmpose.apis import init_model as init_pose_estimator, inference_topdown
        import torch
        
        # Load config
        config_file = Config.fromfile(config_path)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pose_estimator = init_pose_estimator(
            config=config_path, 
            checkpoint=model_path, 
            device=device
        )
        
        # Run inference
        results = inference_topdown(pose_estimator, image_path)
        
        # Extract keypoints
        if isinstance(results, list) and len(results) > 0:
            pred_keypoints = results[0].get('pred_instances', {}).get('keypoints', None)
            if pred_keypoints is not None:
                pred_keypoints = pred_keypoints[0].cpu().numpy()
                if len(pred_keypoints.shape) == 2 and pred_keypoints.shape[1] == 3:
                    pred_keypoints = pred_keypoints[:, :2]
                
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
        print(f"Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("Testing CLdetection2023 Model Accuracy on Aariz Dataset")
    print("="*80)
    
    # Check repository
    cldetection_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CLdetection2023"))
    if not os.path.exists(cldetection_repo):
        print(f"\nERROR: CLdetection2023 repository not found at: {cldetection_repo}")
        print("Please clone it first:")
        print("  git clone https://github.com/5k5000/CLdetection2023.git")
        return
    
    config_path = os.path.join(cldetection_repo, "configs", "CLdetection2023", "srpose_s2.py")
    model_path = os.path.join(cldetection_repo, "model_pretrained_on_train_and_val.pth")
    
    if not os.path.exists(config_path):
        print(f"\nERROR: Config file not found: {config_path}")
        return
    
    if not os.path.exists(model_path):
        # Try in Aariz folder
        model_path = os.path.join(os.path.dirname(__file__), "model_pretrained_on_train_and_val.pth")
        if not os.path.exists(model_path):
            print(f"\nERROR: Model file not found!")
            return
    
    # Check MMPose
    try:
        # Add mmpose to path
        mmpose_path = os.path.join(cldetection_repo, "mmpose_package", "mmpose")
        if os.path.exists(mmpose_path):
            sys.path.insert(0, mmpose_path)
        from mmpose.apis import init_model as init_pose_estimator
        print("\nMMPose is available!")
    except ImportError as e:
        print("\n" + "="*80)
        print("ERROR: MMPose is not installed!")
        print("="*80)
        print("\nPlease install MMPose first:")
        print("  conda create -n LMD python=3.10")
        print("  conda activate LMD")
        print("  pip install -r requirements.txt")
        print("  pip install -U openmim")
        print("  cd mmpose_package/mmpose")
        print("  pip install -e .")
        print("  mim install mmengine")
        print("  mim install 'mmcv>=2.0.0'")
        print("\nThen run this script again.")
        return
    
    # Get test images
    print(f"\nLoading test images from Aariz dataset...")
    test_images = get_test_images("Aariz", num_images=10)
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    # Initialize model
    print(f"\nLoading model...")
    print(f"  Config: {config_path}")
    print(f"  Model: {model_path}")
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from mmengine.config import Config
    from mmpose.apis import init_model as init_pose_estimator, inference_topdown
    
    try:
        pose_estimator = init_pose_estimator(
            config=config_path,
            checkpoint=model_path,
            device=device
        )
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    all_errors_mm = []
    landmark_errors = {lm: [] for lm in COMMON_LANDMARKS}
    successful = 0
    
    print("\nRunning inference on test images...")
    for item in tqdm(test_images):
        image_path = item['image_path']
        image_id = item['image_id']
        gt_landmarks = item['gt_landmarks']
        pixel_size = item['pixel_size']
        
        try:
            # Run inference
            results = inference_topdown(pose_estimator, image_path)
            
            if isinstance(results, list) and len(results) > 0:
                pred_keypoints = results[0].get('pred_instances', {}).get('keypoints', None)
                if pred_keypoints is not None:
                    pred_keypoints = pred_keypoints[0].cpu().numpy()
                    if len(pred_keypoints.shape) == 2 and pred_keypoints.shape[1] == 3:
                        pred_keypoints = pred_keypoints[:, :2]
                    
                    # Map to Aariz landmarks
                    pred_mapped = {}
                    for cld_idx, aariz_lm in CLDETECTION_TO_AARIZ_MAPPING.items():
                        if cld_idx < len(pred_keypoints):
                            pred_mapped[aariz_lm] = {
                                'x': float(pred_keypoints[cld_idx][0]),
                                'y': float(pred_keypoints[cld_idx][1])
                            }
                    
                    successful += 1
                    
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
        
        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate statistics
    if len(all_errors_mm) == 0:
        print("\nNo valid predictions found!")
        return
    
    print("\n" + "="*80)
    print("RESULTS - Common Landmarks Only (15 landmarks)")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Successful predictions: {successful}/{len(test_images)}")
    print(f"  Total landmark predictions: {len(all_errors_mm)}")
    print(f"  Mean Error (MRE): {np.mean(all_errors_mm):.2f} mm")
    print(f"  Median Error: {np.median(all_errors_mm):.2f} mm")
    print(f"  Std Error: {np.std(all_errors_mm):.2f} mm")
    print(f"  Min Error: {np.min(all_errors_mm):.2f} mm")
    print(f"  Max Error: {np.max(all_errors_mm):.2f} mm")
    
    # SDR (Success Detection Rate)
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
    
    # Save results to file
    results_file = "cldetection_accuracy_results.json"
    results = {
        'model': 'CLdetection2023',
        'num_test_images': len(test_images),
        'successful_predictions': successful,
        'total_landmark_predictions': len(all_errors_mm),
        'overall_mre_mm': float(np.mean(all_errors_mm)),
        'overall_median_mm': float(np.median(all_errors_mm)),
        'overall_std_mm': float(np.std(all_errors_mm)),
        'sdr': {
            f'{t}mm': float((np.array(all_errors_mm) <= t).sum() / len(all_errors_mm) * 100)
            for t in thresholds
        },
        'per_landmark': {
            lm: {
                'count': len(landmark_errors[lm]),
                'mre_mm': float(np.mean(landmark_errors[lm])) if len(landmark_errors[lm]) > 0 else None,
                'median_mm': float(np.median(landmark_errors[lm])) if len(landmark_errors[lm]) > 0 else None,
                'sdr_2mm': float((np.array(landmark_errors[lm]) <= 2.0).sum() / len(landmark_errors[lm]) * 100) if len(landmark_errors[lm]) > 0 else None
            }
            for lm in COMMON_LANDMARKS
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

