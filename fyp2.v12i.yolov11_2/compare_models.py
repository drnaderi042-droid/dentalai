"""
اسکریپت مقایسه مدل‌های fyp2 با TTA و بدون TTA
مقایسه با ground truth و تولید گزارش کامل
"""

import os
import sys
import json
import yaml
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
import torch

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# مسیرهای مدل‌ها
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent

# مدل‌های FYP2 (18 کلاس)
FYP2_MODEL_PATHS = [
    BASE_DIR / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt',
    BASE_DIR / 'main_dataset_optimized' / 'phase1_run' / 'weights' / 'best.pt',
    BASE_DIR / 'main_dataset_optimized' / 'phase2_run' / 'weights' / 'best.pt',
]

# مدل‌های Lateral ORTHO (11 کلاس)
# مدل‌های موجود در پوشه fyp2
LATERAL_MODEL_PATHS_FYP2 = [
    BASE_DIR / 'runs' / 'detect' / 'train2' / 'weights' / 'best.pt',
    BASE_DIR / 'optimized_train' / 'run1' / 'weights' / 'best.pt',
]

# مدل‌های موجود در پوشه اصلی LATERAL ORTHO AI.v2i.yolov8
LATERAL_MODEL_PATHS_MAIN = [
    PARENT_DIR / 'LATERAL ORTHO AI.v2i.yolov8' / 'runs' / 'detect' / 'train2' / 'weights' / 'best.pt',
    PARENT_DIR / 'LATERAL ORTHO AI.v2i.yolov8' / 'runs' / 'detect' / 'ortho_improved' / 'weights' / 'best.pt',
]

# ترکیب تمام مدل‌های Lateral
LATERAL_MODEL_PATHS = LATERAL_MODEL_PATHS_FYP2 + LATERAL_MODEL_PATHS_MAIN

# مسیرهای دیتاست FYP2 (18 کلاس)
FYP2_DATA_YAML = BASE_DIR / 'data.yaml'
FYP2_TEST_IMAGES_DIR = BASE_DIR / 'test' / 'images'
FYP2_TEST_LABELS_DIR = BASE_DIR / 'test' / 'labels'

# مسیرهای دیتاست Lateral ORTHO (11 کلاس)
# اول بررسی می‌کنیم که آیا در پوشه fyp2 وجود دارد
LATERAL_DATA_YAML_FYP2 = BASE_DIR / 'LATERAL ORTHO AI.v2i.yolov11' / 'data.yaml'
LATERAL_TEST_IMAGES_DIR_FYP2 = BASE_DIR / 'LATERAL ORTHO AI.v2i.yolov11' / 'test' / 'images'
LATERAL_TEST_LABELS_DIR_FYP2 = BASE_DIR / 'LATERAL ORTHO AI.v2i.yolov11' / 'test' / 'labels'

# سپس در پوشه اصلی بررسی می‌کنیم
LATERAL_DATA_YAML_MAIN = PARENT_DIR / 'LATERAL ORTHO AI.v2i.yolov8' / 'data.yaml'
LATERAL_TEST_IMAGES_DIR_MAIN = PARENT_DIR / 'LATERAL ORTHO AI.v2i.yolov8' / 'test' / 'images'
LATERAL_TEST_LABELS_DIR_MAIN = PARENT_DIR / 'LATERAL ORTHO AI.v2i.yolov8' / 'test' / 'labels'

# انتخاب مسیرهای Lateral (اول FYP2، سپس main)
if LATERAL_TEST_IMAGES_DIR_FYP2.exists():
    LATERAL_DATA_YAML = LATERAL_DATA_YAML_FYP2
    LATERAL_TEST_IMAGES_DIR = LATERAL_TEST_IMAGES_DIR_FYP2
    LATERAL_TEST_LABELS_DIR = LATERAL_TEST_LABELS_DIR_FYP2
    print(f"📁 Using Lateral ORTHO dataset from: {LATERAL_TEST_IMAGES_DIR_FYP2}")
elif LATERAL_TEST_IMAGES_DIR_MAIN.exists():
    LATERAL_DATA_YAML = LATERAL_DATA_YAML_MAIN
    LATERAL_TEST_IMAGES_DIR = LATERAL_TEST_IMAGES_DIR_MAIN
    LATERAL_TEST_LABELS_DIR = LATERAL_TEST_LABELS_DIR_MAIN
    print(f"📁 Using Lateral ORTHO dataset from: {LATERAL_TEST_IMAGES_DIR_MAIN}")
else:
    LATERAL_DATA_YAML = None
    LATERAL_TEST_IMAGES_DIR = None
    LATERAL_TEST_LABELS_DIR = None
    print(f"⚠️ Lateral ORTHO test dataset not found!")

# تنظیمات TTA
NUM_AUGMENTATIONS = 4  # تعداد augmentation برای TTA

def load_class_names(data_yaml_path):
    """بارگذاری نام کلاس‌ها از data.yaml"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except Exception as e:
        print(f"⚠️ Error loading class names: {e}")
        return []

def parse_yolo_label(label_path):
    """پارس کردن فایل label YOLO format"""
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                    })
    except Exception as e:
        print(f"⚠️ Error parsing label {label_path}: {e}")
    return boxes

def convert_yolo_to_xyxy(box, img_width, img_height):
    """تبدیل YOLO format (normalized center, width, height) به xyxy"""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """محاسبه IoU بین دو bounding box"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def predict_with_tta(model, image_path, conf_threshold=0.001, num_augmentations=NUM_AUGMENTATIONS):
    """پیش‌بینی با Test-Time Augmentation"""
    results_list = []
    
    # پیش‌بینی بدون augmentation
    results = model.predict(image_path, conf=conf_threshold, verbose=False, augment=False)
    if results and len(results) > 0:
        results_list.append(results[0])
    
    # پیش‌بینی با augmentation (TTA)
    for _ in range(num_augmentations):
        results = model.predict(image_path, conf=conf_threshold, verbose=False, augment=True)
        if results and len(results) > 0:
            results_list.append(results[0])
    
    # ترکیب نتایج (میانگین confidence و bounding boxes)
    if not results_list:
        return []
    
    # استخراج تمام detections
    all_detections = []
    for result in results_list:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls.astype(int)
            
            for i in range(len(boxes)):
                all_detections.append({
                    'box': boxes[i].tolist(),
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                })
    
    # Group by class and average
    detections_by_class = defaultdict(list)
    for det in all_detections:
        detections_by_class[det['class_id']].append(det)
    
    # Average detections for each class
    final_detections = []
    for class_id, dets in detections_by_class.items():
        if len(dets) > 0:
            # Average confidence
            avg_confidence = np.mean([d['confidence'] for d in dets])
            # Average box (weighted by confidence)
            weights = [d['confidence'] for d in dets]
            weights = np.array(weights) / np.sum(weights)
            
            avg_box = np.zeros(4)
            for i, det in enumerate(dets):
                avg_box += weights[i] * np.array(det['box'])
            
            final_detections.append({
                'class_id': class_id,
                'confidence': float(avg_confidence),
                'box': avg_box.tolist(),
            })
    
    return final_detections

def evaluate_model_with_yolo_validation(model_path, data_yaml_path, use_tta=False, conf_threshold=0.25, iou_threshold=0.5):
    """ارزیابی مدل با استفاده از متد validation خود YOLO (مانند training)"""
    print(f"\n{'='*80}")
    print(f"🔍 Evaluating model with YOLO validation: {model_path.name}")
    print(f"   Method: YOLO standard validation (same as training)")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {iou_threshold}")
    print(f"{'='*80}")
    
    # بارگذاری مدل
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # بررسی نام کلاس‌های مدل
    if hasattr(model, 'names'):
        model_class_names = list(model.names.values())
        print(f"📋 Model classes ({len(model_class_names)}): {model_class_names[:5]}...")
    
    # استفاده از متد validation YOLO (مانند training)
    try:
        print(f"🔄 Running YOLO validation (this uses the same method as training)...")
        
        # اجرای validation با تنظیمات مشابه training
        results = model.val(
            data=str(data_yaml_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True,
            plots=False,
            save_json=False,
        )
        
        # استخراج معیارها از نتایج YOLO
        if hasattr(results, 'box'):
            metrics = {
                'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                'mAP50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
                'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            }
        else:
            print(f"⚠️ Could not extract metrics from results")
            return None
        
        print(f"✅ Validation completed!")
        print(f"   mAP50: {metrics['mAP50']:.4f}")
        print(f"   mAP50-95: {metrics['mAP50_95']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        # محاسبه F1 Score
        f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        
        return {
            'mAP50': metrics['mAP50'],
            'mAP50_95': metrics['mAP50_95'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': f1_score,
            'use_yolo_validation': True,
        }
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def evaluate_model(model_path, test_images_dir, test_labels_dir, class_names, use_tta=False, conf_threshold=0.001, iou_threshold=0.5):
    """ارزیابی یک مدل روی test set"""
    print(f"\n{'='*80}")
    print(f"🔍 Evaluating model: {model_path.name}")
    print(f"   TTA: {'Enabled' if use_tta else 'Disabled'}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {iou_threshold}")
    print(f"{'='*80}")
    
    # بارگذاری مدل
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # بررسی نام کلاس‌های مدل
    if hasattr(model, 'names'):
        model_class_names = list(model.names.values())
        print(f"📋 Model classes ({len(model_class_names)}): {model_class_names[:5]}...")
    
    # آمارها
    stats = {
        'total_images': 0,
        'total_gt_boxes': 0,
        'total_pred_boxes': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_class_tp': defaultdict(int),
        'per_class_fp': defaultdict(int),
        'per_class_fn': defaultdict(int),
        'processing_times': [],
        'confidences': [],
    }
    
    # پردازش تمام تصاویر تست
    test_images = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
    print(f"📸 Processing {len(test_images)} test images...")
    
    for idx, image_path in enumerate(test_images):
        if (idx + 1) % 20 == 0:
            print(f"   Processed {idx + 1}/{len(test_images)} images...")
        
        # بارگذاری ground truth
        label_path = test_labels_dir / f"{image_path.stem}.txt"
        gt_boxes = parse_yolo_label(label_path)
        
        # بارگذاری تصویر برای ابعاد
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️ Error loading image {image_path}: {e}")
            continue
        
        # تبدیل GT boxes به xyxy
        gt_boxes_xyxy = []
        for gt_box in gt_boxes:
            xyxy = convert_yolo_to_xyxy(gt_box, img_width, img_height)
            gt_boxes_xyxy.append({
                'box': xyxy,
                'class_id': gt_box['class_id'],
            })
        
        # پیش‌بینی
        start_time = time.time()
        if use_tta:
            pred_detections = predict_with_tta(model, str(image_path), conf_threshold)
        else:
            results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
            pred_detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy
                    confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf
                    class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls.astype(int)
                    
                    for i in range(len(boxes)):
                        pred_detections.append({
                            'box': boxes[i].tolist(),
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i]),
                        })
        processing_time = time.time() - start_time
        
        stats['total_images'] += 1
        stats['total_gt_boxes'] += len(gt_boxes_xyxy)
        stats['total_pred_boxes'] += len(pred_detections)
        stats['processing_times'].append(processing_time)
        stats['confidences'].extend([d['confidence'] for d in pred_detections])
        
        # تطبیق detections با ground truth
        matched_gt = set()
        matched_pred = set()
        
        for pred_idx, pred in enumerate(pred_detections):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes_xyxy):
                if gt_idx in matched_gt:
                    continue
                
                if pred['class_id'] == gt['class_id']:
                    iou = calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True Positive
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                stats['true_positives'] += 1
                stats['per_class_tp'][pred['class_id']] += 1
            else:
                # False Positive
                stats['false_positives'] += 1
                stats['per_class_fp'][pred['class_id']] += 1
        
        # False Negatives
        for gt_idx, gt in enumerate(gt_boxes_xyxy):
            if gt_idx not in matched_gt:
                stats['false_negatives'] += 1
                stats['per_class_fn'][gt['class_id']] += 1
    
    # محاسبه معیارها
    precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (stats['true_positives'] + stats['false_positives']) > 0 else 0
    recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives']) if (stats['true_positives'] + stats['false_negatives']) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats['precision'] = precision
    stats['recall'] = recall
    stats['f1_score'] = f1_score
    stats['avg_processing_time'] = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    stats['avg_confidence'] = np.mean(stats['confidences']) if stats['confidences'] else 0
    
    return stats

def generate_report(all_results, output_file='model_comparison_report.md'):
    """تولید گزارش مقایسه مدل‌ها"""
    report = ["# 📊 گزارش مقایسه مدل‌های FYP2 و Lateral ORTHO\n"]
    report.append(f"تاریخ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n\n")
    report.append("## 📝 توضیحات\n\n")
    report.append("- **FYP2 Models**: مدل‌های 18 کلاس (canine/molar Class I/II/III با subdivisions)\n")
    report.append("- **Lateral Models**: مدل‌های 11 کلاس (Class I/II/III + مشکلات دندانی)\n")
    report.append("- هر مدل فقط با dataset مربوط به خودش مقایسه شده است\n\n")
    report.append("---\n\n")
    
    # جدول مقایسه FYP2 (YOLO validation)
    fyp2_yolo_results = [r for r in all_results if r.get('dataset_type') == 'fyp2' and r.get('use_yolo_validation')]
    if fyp2_yolo_results:
        report.append("## 📈 جدول مقایسه مدل‌های FYP2 (18 کلاس) - YOLO Validation (استاندارد)\n\n")
        report.append("| Model | mAP50 | mAP50-95 | Precision | Recall | F1 Score |\n")
        report.append("|-------|-------|----------|-----------|--------|----------|\n")
        
        for result in fyp2_yolo_results:
            model_name = result['model_name'].replace(' (FYP2, YOLO val)', '').replace('FYP2, ', '')
            stats = result['stats']
            
            report.append(f"| {model_name} | "
                         f"{stats.get('mAP50', 0):.4f} | {stats.get('mAP50_95', 0):.4f} | "
                         f"{stats.get('precision', 0):.4f} | {stats.get('recall', 0):.4f} | "
                         f"{stats.get('f1_score', 0):.4f} |\n")
        
        report.append("\n---\n\n")
    
    # جدول مقایسه FYP2 (Manual - برای مقایسه)
    fyp2_manual_results = [r for r in all_results if r.get('dataset_type') == 'fyp2' and not r.get('use_yolo_validation')]
    if fyp2_manual_results:
        report.append("## 📈 جدول مقایسه مدل‌های FYP2 (18 کلاس) - Manual Method (برای مقایسه)\n\n")
        report.append("| Model | TTA | Precision | Recall | F1 Score | Avg Time (s) | Avg Confidence |\n")
        report.append("|-------|-----|-----------|--------|----------|--------------|----------------|\n")
        
        for result in fyp2_manual_results:
            model_name = result['model_name'].replace(' (FYP2, manual, no TTA)', '').replace('FYP2, ', '')
            use_tta = result['use_tta']
            stats = result['stats']
            
            report.append(f"| {model_name} | {'✅' if use_tta else '❌'} | "
                         f"{stats.get('precision', 0):.4f} | {stats.get('recall', 0):.4f} | "
                         f"{stats.get('f1_score', 0):.4f} | {stats.get('avg_processing_time', 0):.3f} | "
                         f"{stats.get('avg_confidence', 0):.4f} |\n")
        
        report.append("\n---\n\n")
    
    # جدول مقایسه Lateral (YOLO validation)
    lateral_yolo_results = [r for r in all_results if r.get('dataset_type') == 'lateral' and r.get('use_yolo_validation')]
    if lateral_yolo_results:
        report.append("## 📈 جدول مقایسه مدل‌های Lateral ORTHO (11 کلاس) - YOLO Validation (استاندارد)\n\n")
        report.append("| Model | mAP50 | mAP50-95 | Precision | Recall | F1 Score |\n")
        report.append("|-------|-------|----------|-----------|--------|----------|\n")
        
        for result in lateral_yolo_results:
            model_name = result['model_name'].replace(' (Lateral, YOLO val)', '').replace('Lateral, ', '')
            stats = result['stats']
            
            report.append(f"| {model_name} | "
                         f"{stats.get('mAP50', 0):.4f} | {stats.get('mAP50_95', 0):.4f} | "
                         f"{stats.get('precision', 0):.4f} | {stats.get('recall', 0):.4f} | "
                         f"{stats.get('f1_score', 0):.4f} |\n")
        
        report.append("\n---\n\n")
    
    # جدول مقایسه Lateral (Manual - برای مقایسه)
    lateral_manual_results = [r for r in all_results if r.get('dataset_type') == 'lateral' and not r.get('use_yolo_validation')]
    if lateral_manual_results:
        report.append("## 📈 جدول مقایسه مدل‌های Lateral ORTHO (11 کلاس) - Manual Method (برای مقایسه)\n\n")
        report.append("| Model | TTA | Precision | Recall | F1 Score | Avg Time (s) | Avg Confidence |\n")
        report.append("|-------|-----|-----------|--------|----------|--------------|----------------|\n")
        
        for result in lateral_manual_results:
            model_name = result['model_name'].replace(' (Lateral, manual, no TTA)', '').replace('Lateral, ', '')
            use_tta = result['use_tta']
            stats = result['stats']
            
            report.append(f"| {model_name} | {'✅' if use_tta else '❌'} | "
                         f"{stats.get('precision', 0):.4f} | {stats.get('recall', 0):.4f} | "
                         f"{stats.get('f1_score', 0):.4f} | {stats.get('avg_processing_time', 0):.3f} | "
                         f"{stats.get('avg_confidence', 0):.4f} |\n")
        
        report.append("\n---\n\n")
    
    # جزئیات هر مدل
    report.append("## 🔍 جزئیات هر مدل\n\n")
    
    for result in all_results:
        model_name = result['model_name']
        use_tta = result['use_tta']
        stats = result['stats']
        
        report.append(f"### {model_name} {'(with TTA)' if use_tta else '(without TTA)'}\n\n")
        report.append(f"- **Precision**: {stats['precision']:.4f}\n")
        report.append(f"- **Recall**: {stats['recall']:.4f}\n")
        report.append(f"- **F1 Score**: {stats['f1_score']:.4f}\n")
        report.append(f"- **True Positives**: {stats['true_positives']}\n")
        report.append(f"- **False Positives**: {stats['false_positives']}\n")
        report.append(f"- **False Negatives**: {stats['false_negatives']}\n")
        report.append(f"- **Average Processing Time**: {stats['avg_processing_time']:.3f}s\n")
        report.append(f"- **Average Confidence**: {stats['avg_confidence']:.4f}\n")
        report.append(f"- **Total Images**: {stats['total_images']}\n")
        report.append(f"- **Total GT Boxes**: {stats['total_gt_boxes']}\n")
        report.append(f"- **Total Pred Boxes**: {stats['total_pred_boxes']}\n\n")
        
        report.append("---\n\n")
    
    # ذخیره گزارش
    output_path = BASE_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"\n✅ گزارش ذخیره شد: {output_path}")

def detect_model_dataset_type(model_path):
    """تشخیص نوع dataset مدل (FYP2 یا Lateral) بر اساس کلاس‌های مدل"""
    try:
        model = YOLO(str(model_path))
        if hasattr(model, 'names'):
            class_names = list(model.names.values())
            num_classes = len(class_names)
            
            # بررسی اینکه آیا مدل FYP2 است (18 کلاس با canine/molar)
            if num_classes == 18:
                if any('canine' in name.lower() or 'molar' in name.lower() for name in class_names[:5]):
                    return 'fyp2', num_classes, class_names
            
            # بررسی اینکه آیا مدل Lateral است (11 کلاس با Class I/II/III)
            if num_classes == 11:
                if any('class i' in name.lower() or 'class ii' in name.lower() or 'class iii' in name.lower() for name in class_names[:3]):
                    return 'lateral', num_classes, class_names
            
            # اگر نتوانستیم تشخیص دهیم، بر اساس تعداد کلاس حدس می‌زنیم
            if num_classes == 18:
                return 'fyp2', num_classes, class_names
            elif num_classes == 11:
                return 'lateral', num_classes, class_names
            else:
                return 'unknown', num_classes, class_names
    except Exception as e:
        print(f"⚠️ Error detecting model type: {e}")
        return 'unknown', 0, []
    
    return 'unknown', 0, []

def main():
    """تابع اصلی"""
    print("🚀 شروع مقایسه مدل‌های FYP2 و Lateral ORTHO\n")
    
    # بارگذاری نام کلاس‌ها برای هر dataset
    fyp2_class_names = load_class_names(FYP2_DATA_YAML) if FYP2_DATA_YAML.exists() else []
    lateral_class_names = load_class_names(LATERAL_DATA_YAML) if LATERAL_DATA_YAML and LATERAL_DATA_YAML.exists() else []
    
    print(f"📋 FYP2 Dataset: {len(fyp2_class_names)} classes")
    if fyp2_class_names:
        print(f"   Classes: {fyp2_class_names[:5]}...\n")
    
    print(f"📋 Lateral ORTHO Dataset: {len(lateral_class_names)} classes")
    if lateral_class_names:
        print(f"   Classes: {lateral_class_names[:5]}...\n")
    
    # بررسی وجود مدل‌ها و تشخیص نوع dataset
    fyp2_models = []
    lateral_models = []
    
    print("🔍 بررسی مدل‌های FYP2 (18 کلاس):")
    for model_path in FYP2_MODEL_PATHS:
        if model_path.exists():
            dataset_type, num_classes, class_names = detect_model_dataset_type(model_path)
            print(f"   ✅ {model_path.name}: {num_classes} classes, type: {dataset_type}")
            if dataset_type == 'fyp2':
                fyp2_models.append(model_path)
            else:
                print(f"      ⚠️ Warning: Expected FYP2 but detected {dataset_type}")
        else:
            print(f"   ⚠️ Not found: {model_path}")
    
    print("\n🔍 بررسی مدل‌های Lateral ORTHO (11 کلاس):")
    for model_path in LATERAL_MODEL_PATHS:
        if model_path.exists():
            dataset_type, num_classes, class_names = detect_model_dataset_type(model_path)
            print(f"   ✅ {model_path.name}: {num_classes} classes, type: {dataset_type}")
            if dataset_type == 'lateral':
                lateral_models.append(model_path)
            else:
                print(f"      ⚠️ Warning: Expected Lateral but detected {dataset_type}")
        else:
            print(f"   ⚠️ Not found: {model_path}")
    
    if not fyp2_models and not lateral_models:
        print("\n❌ No models found!")
        return
    
    print(f"\n📊 تعداد مدل‌های FYP2: {len(fyp2_models)}")
    print(f"📊 تعداد مدل‌های Lateral: {len(lateral_models)}\n")
    
    # ارزیابی مدل‌ها
    all_results = []
    
    # ارزیابی مدل‌های FYP2
    if fyp2_models and FYP2_TEST_IMAGES_DIR.exists():
        print(f"\n{'='*80}")
        print(f"📊 ارزیابی مدل‌های FYP2 (18 کلاس)")
        print(f"{'='*80}\n")
        
        for model_path in fyp2_models:
            # استخراج نام مدل از مسیر
            if 'runs' in str(model_path):
                model_name = model_path.parent.parent.name
            elif 'main_dataset_optimized' in str(model_path):
                model_name = f"{model_path.parent.parent.name}_{model_path.parent.name}"
            else:
                model_name = model_path.parent.parent.parent.name
            
            # استفاده از YOLO validation (روش استاندارد - مانند training)
            print(f"\n{'='*80}")
            print(f"🔄 Evaluating {model_name} (FYP2, YOLO validation - standard method)...")
            stats_yolo = evaluate_model_with_yolo_validation(
                model_path, FYP2_DATA_YAML,
                use_tta=False, conf_threshold=0.25, iou_threshold=0.5
            )
            
            if stats_yolo:
                all_results.append({
                    'model_name': f"{model_name} (FYP2, YOLO val)",
                    'model_path': str(model_path),
                    'dataset_type': 'fyp2',
                    'use_tta': False,
                    'use_yolo_validation': True,
                    'stats': stats_yolo,
                })
            
            # همچنین ارزیابی دستی (برای مقایسه)
            print(f"\n{'='*80}")
            print(f"🔄 Evaluating {model_name} (FYP2, manual method - for comparison)...")
            stats_no_tta = evaluate_model(
                model_path, FYP2_TEST_IMAGES_DIR, FYP2_TEST_LABELS_DIR, fyp2_class_names,
                use_tta=False, conf_threshold=0.25, iou_threshold=0.5  # استفاده از threshold مناسب
            )
            
            if stats_no_tta:
                all_results.append({
                    'model_name': f"{model_name} (FYP2, manual, no TTA)",
                    'model_path': str(model_path),
                    'dataset_type': 'fyp2',
                    'use_tta': False,
                    'use_yolo_validation': False,
                    'stats': stats_no_tta,
                })
    else:
        if not fyp2_models:
            print("⚠️ No FYP2 models found!")
        if not FYP2_TEST_IMAGES_DIR.exists():
            print(f"⚠️ FYP2 test dataset not found: {FYP2_TEST_IMAGES_DIR}")
    
    # ارزیابی مدل‌های Lateral
    if lateral_models and LATERAL_TEST_IMAGES_DIR and LATERAL_TEST_IMAGES_DIR.exists():
        print(f"\n{'='*80}")
        print(f"📊 ارزیابی مدل‌های Lateral ORTHO (11 کلاس)")
        print(f"{'='*80}\n")
        
        for model_path in lateral_models:
            # استخراج نام مدل از مسیر
            if 'runs' in str(model_path):
                if 'LATERAL ORTHO AI.v2i.yolov8' in str(model_path):
                    # مدل از پوشه اصلی lateral
                    model_name = f"lateral_main_{model_path.parent.parent.name}"
                else:
                    # مدل از پوشه fyp2
                    model_name = f"lateral_fyp2_{model_path.parent.parent.name}"
            elif 'optimized_train' in str(model_path):
                model_name = f"lateral_fyp2_{model_path.parent.parent.name}"
            else:
                model_name = model_path.parent.parent.parent.name
            
            # استفاده از YOLO validation (روش استاندارد - مانند training)
            if LATERAL_DATA_YAML and LATERAL_DATA_YAML.exists():
                print(f"\n{'='*80}")
                print(f"🔄 Evaluating {model_name} (Lateral, YOLO validation - standard method)...")
                stats_yolo = evaluate_model_with_yolo_validation(
                    model_path, LATERAL_DATA_YAML,
                    use_tta=False, conf_threshold=0.25, iou_threshold=0.5
                )
                
                if stats_yolo:
                    all_results.append({
                        'model_name': f"{model_name} (Lateral, YOLO val)",
                        'model_path': str(model_path),
                        'dataset_type': 'lateral',
                        'use_tta': False,
                        'use_yolo_validation': True,
                        'stats': stats_yolo,
                    })
            
            # همچنین ارزیابی دستی (برای مقایسه)
            print(f"\n{'='*80}")
            print(f"🔄 Evaluating {model_name} (Lateral, manual method - for comparison)...")
            stats_no_tta = evaluate_model(
                model_path, LATERAL_TEST_IMAGES_DIR, LATERAL_TEST_LABELS_DIR, lateral_class_names,
                use_tta=False, conf_threshold=0.25, iou_threshold=0.5  # استفاده از threshold مناسب
            )
            
            if stats_no_tta:
                all_results.append({
                    'model_name': f"{model_name} (Lateral, manual, no TTA)",
                    'model_path': str(model_path),
                    'dataset_type': 'lateral',
                    'use_tta': False,
                    'use_yolo_validation': False,
                    'stats': stats_no_tta,
                })
    else:
        if not lateral_models:
            print("⚠️ No Lateral models found!")
        if not LATERAL_TEST_IMAGES_DIR or not LATERAL_TEST_IMAGES_DIR.exists():
            print(f"⚠️ Lateral test dataset not found!")
    
    # تولید گزارش
    if all_results:
        generate_report(all_results)
        print(f"\n✅ مقایسه کامل شد! {len(all_results)} نتیجه تولید شد.")
        
        # خلاصه
        fyp2_results = [r for r in all_results if r['dataset_type'] == 'fyp2']
        lateral_results = [r for r in all_results if r['dataset_type'] == 'lateral']
        print(f"   - FYP2 models: {len(fyp2_results)} results")
        print(f"   - Lateral models: {len(lateral_results)} results")
    else:
        print("\n❌ هیچ نتیجه‌ای تولید نشد!")

if __name__ == '__main__':
    main()

