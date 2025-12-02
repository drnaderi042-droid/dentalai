"""
Test calibration point detection for ruler markers in cephalometric radiographs.
This script tests the detection of p1 and p2 points on the ruler.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

def load_aariz_annotation(json_path):
    """Load Aariz dataset annotation."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_ground_truth(annotation_data):
    """Extract p1 and p2 ground truth from Aariz annotation."""
    points = {}
    
    if 'landmarks' not in annotation_data:
        return None
    
    landmarks = annotation_data['landmarks']
    
    for landmark in landmarks:
        symbol = landmark.get('symbol', '')
        if symbol in ['p1', 'p2']:
            value = landmark.get('value', {})
            points[symbol] = {
                'x': int(value['x']),
                'y': int(value['y'])
            }
    
    # Check if both p1 and p2 were found
    if 'p1' in points and 'p2' in points:
        return points
    
    return None

def detect_bright_calibration_points(image_path, visualize=False):
    """
    Detect bright calibration points in upper-right corner of image.
    Returns list of detected points sorted by brightness.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return []
    
    height, width = img.shape[:2]
    
    # Define search area (upper right 20%)
    search_x_start = int(width * 0.7)  # Start from 70% width
    search_y_end = int(height * 0.25)  # Search top 25%
    
    # Extract search region
    search_region = img[0:search_y_end, search_x_start:width]
    
    # Convert to grayscale
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    
    # Find bright spots - lower threshold for better detection
    # Using 150 instead of 180 to catch dimmer ruler marks
    _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright regions
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get centroids and brightness of each bright region
    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10 and area < 500:  # Filter by reasonable size
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Calculate average brightness in this region
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                brightness = cv2.mean(gray, mask=mask)[0]
                
                # Convert to image coordinates
                img_x = search_x_start + cx
                img_y = cy
                
                points.append({
                    'x': img_x,
                    'y': img_y,
                    'brightness': brightness,
                    'area': area
                })
    
    # Sort by brightness (brightest first)
    points.sort(key=lambda p: p['brightness'], reverse=True)
    
    # Filter: find two points that are vertically aligned
    calibration_pair = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            
            dx = abs(p1['x'] - p2['x'])
            dy = abs(p1['y'] - p2['y'])
            
            # Check if vertically aligned and separated
            if dx < 30 and 70 < dy < 150:
                # Determine which is upper (p2) and lower (p1)
                upper = p1 if p1['y'] < p2['y'] else p2
                lower = p2 if p1['y'] < p2['y'] else p1
                
                calibration_pair = [upper, lower]  # [p2, p1]
                break
        
        if calibration_pair:
            break
    
    if visualize:
        # Draw all detected points
        vis_img = img.copy()
        for i, p in enumerate(points[:10]):  # Show top 10
            color = (0, 255, 0) if i < 2 and not calibration_pair else (255, 0, 0)
            cv2.circle(vis_img, (p['x'], p['y']), 5, color, -1)
            cv2.putText(vis_img, f"{i+1}", (p['x']+10, p['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw calibration pair if found
        if calibration_pair:
            p2, p1 = calibration_pair
            cv2.circle(vis_img, (p2['x'], p2['y']), 8, (0, 0, 255), 2)
            cv2.circle(vis_img, (p1['x'], p1['y']), 8, (0, 0, 255), 2)
            cv2.line(vis_img, (p2['x'], p2['y']), (p1['x'], p1['y']), (0, 0, 255), 2)
            
            distance = np.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
            mm_per_pixel = 10 / distance
            
            cv2.putText(vis_img, f"p2 (upper)", (p2['x']+15, p2['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis_img, f"p1 (lower)", (p1['x']+15, p1['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis_img, f"Dist: {distance:.1f}px", (p2['x']+15, p2['y']+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis_img, f"Conv: {mm_per_pixel:.4f} mm/px", (p2['x']+15, p2['y']+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw search area
        cv2.rectangle(vis_img, (search_x_start, 0), (width, search_y_end), (255, 255, 0), 2)
        
        return calibration_pair, vis_img
    
    return calibration_pair

def test_calibration_detection():
    """Test calibration detection on all images."""
    # Use the main Aariz dataset
    annotations_dir = Path('Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists')
    images_dir = Path('Aariz/train/Cephalograms')
    
    if not annotations_dir.exists():
        print(f"❌ Annotations directory not found: {annotations_dir}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Get all annotation files that match our P1P2 images
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
    
    print(f"📊 Testing {len(p1p2_image_ids)} images with P1/P2 annotations")
    
    results = []
    correct_detections = 0
    
    # Create output directory for visualizations
    output_dir = Path('calibration_test_results')
    output_dir.mkdir(exist_ok=True)
    
    for i, image_id in enumerate(p1p2_image_ids):
        # Get annotation file
        annotation_file = annotations_dir / f"{image_id}.json"
        
        if not annotation_file.exists():
            print(f"⚠️ Annotation not found: {annotation_file}")
            continue
        
        # Load annotation
        annotation_data = load_aariz_annotation(annotation_file)
        
        # Get ground truth
        ground_truth = get_ground_truth(annotation_data)
        if not ground_truth:
            print(f"⚠️ No p1/p2 in annotation: {image_id}")
            continue
        
        # Find image file (could be .png, .jpg, or .jpeg)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = images_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if not image_path:
            print(f"⚠️ Image not found for: {image_id}")
            continue
        
        filename = image_path.name
        
        gt_p1 = ground_truth['p1']
        gt_p2 = ground_truth['p2']
        
        # Detect calibration points
        detected_pair, vis_img = detect_bright_calibration_points(image_path, visualize=True)
        
        if detected_pair and len(detected_pair) == 2:
            det_p2, det_p1 = detected_pair
            
            # Calculate error
            error_p1 = np.sqrt((det_p1['x'] - gt_p1['x'])**2 + (det_p1['y'] - gt_p1['y'])**2)
            error_p2 = np.sqrt((det_p2['x'] - gt_p2['x'])**2 + (det_p2['y'] - gt_p2['y'])**2)
            
            # Calculate distance
            gt_distance = np.sqrt((gt_p1['x'] - gt_p2['x'])**2 + (gt_p1['y'] - gt_p2['y'])**2)
            det_distance = np.sqrt((det_p1['x'] - det_p2['x'])**2 + (det_p1['y'] - det_p2['y'])**2)
            
            # Check if detection is correct (within 20 pixels)
            is_correct = error_p1 < 20 and error_p2 < 20
            if is_correct:
                correct_detections += 1
            
            result = {
                'filename': filename,
                'ground_truth': ground_truth,
                'detected': {'p1': {'x': det_p1['x'], 'y': det_p1['y']}, 'p2': {'x': det_p2['x'], 'y': det_p2['y']}},
                'error_p1': error_p1,
                'error_p2': error_p2,
                'gt_distance': gt_distance,
                'det_distance': det_distance,
                'mm_per_pixel': 10 / det_distance,
                'correct': is_correct
            }
            results.append(result)
            
            # Draw ground truth on visualization
            cv2.circle(vis_img, (gt_p2['x'], gt_p2['y']), 6, (0, 255, 255), 2)
            cv2.circle(vis_img, (gt_p1['x'], gt_p1['y']), 6, (0, 255, 255), 2)
            cv2.putText(vis_img, "GT", (gt_p2['x']-25, gt_p2['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            status_color = (0, 255, 0) if is_correct else (0, 0, 255)
            status_text = f"{'PASS' if is_correct else 'FAIL'} - Error: p1={error_p1:.1f}px, p2={error_p2:.1f}px"
            cv2.putText(vis_img, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Save visualization
            output_path = output_dir / f"{i+1:02d}_{filename}"
            cv2.imwrite(str(output_path), vis_img)
            
            print(f"✅ {filename}: p1_error={error_p1:.1f}px, p2_error={error_p2:.1f}px - {'PASS' if is_correct else 'FAIL'}")
        else:
            print(f"❌ {filename}: No calibration pair detected")
            # Save failed detection
            if vis_img is not None:
                output_path = output_dir / f"{i+1:02d}_{filename}_FAILED.png"
                cv2.imwrite(str(output_path), vis_img)
            results.append({'filename': filename, 'correct': False, 'detected': None})
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(p1p2_image_ids)}")
    print(f"Successful detections: {len([r for r in results if r.get('detected')])}")
    print(f"Correct detections (< 20px error): {correct_detections}/{len(results)}")
    if len(results) > 0:
        print(f"Accuracy: {correct_detections/len(results)*100:.1f}%")
    else:
        print(f"Accuracy: 0%")
    
    if results:
        avg_error_p1 = np.mean([r['error_p1'] for r in results if 'error_p1' in r])
        avg_error_p2 = np.mean([r['error_p2'] for r in results if 'error_p2' in r])
        print(f"\nAverage error:")
        print(f"  p1: {avg_error_p1:.2f} pixels")
        print(f"  p2: {avg_error_p2:.2f} pixels")
        
        avg_mm_per_pixel = np.mean([r['mm_per_pixel'] for r in results if 'mm_per_pixel' in r])
        print(f"\nAverage conversion: {avg_mm_per_pixel:.4f} mm/pixel")
    
    print(f"\n💾 Visualizations saved to: {output_dir}")

if __name__ == '__main__':
    import os
    
    # Get script directory and change to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📂 Working directory: {os.getcwd()}\n")
    
    test_calibration_detection()

