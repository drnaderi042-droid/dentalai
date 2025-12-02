#!/usr/bin/env python3
"""
Dataset Analysis Script for YOLOv11 Lateral Orthodontic AI
Analyzes class distribution, image counts, and identifies potential issues
"""

import os
import json
from collections import Counter
from pathlib import Path

def analyze_dataset(data_path):
    """Analyze the YOLO dataset structure and class distribution"""
    
    print("=== YOLOv11 Lateral Orthodontic AI Dataset Analysis ===\n")
    
    # Define paths
    train_labels = Path(data_path) / "train" / "labels"
    val_labels = Path(data_path) / "valid" / "labels"
    test_labels = Path(data_path) / "test" / "labels"
    
    # Class names (from data.yaml)
    class_names = [
        'Class I', 'Class II', 'Class III', 'Cross bite', 'Crowding', 
        'Deep bite', 'Open bite', 'Proclination', 'Retroclination', 
        'Rotation', 'Spacing'
    ]
    
    def count_annotations(labels_path):
        """Count annotations in a label directory"""
        class_counts = Counter()
        total_annotations = 0
        
        if not labels_path.exists():
            print(f"Warning: {labels_path} does not exist")
            return class_counts, 0
            
        label_files = list(labels_path.glob("*.txt"))
        print(f"Found {len(label_files)} label files in {labels_path.name}")
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_annotations += 1
        
        return class_counts, total_annotations
    
    # Analyze each split
    splits = [
        ("Training", train_labels),
        ("Validation", val_labels), 
        ("Test", test_labels)
    ]
    
    all_counts = {}
    total_images = 0
    
    for split_name, labels_path in splits:
        print(f"\n--- {split_name} Split ---")
        class_counts, total_annotations = count_annotations(labels_path)
        all_counts[split_name] = class_counts
        
        # Count images
        if labels_path.exists():
            image_count = len(list(labels_path.glob("*.txt")))
            total_images += image_count
            print(f"Images: {image_count}")
            print(f"Total annotations: {total_annotations}")
            
            # Class distribution
            print("Class distribution:")
            for i, class_name in enumerate(class_names):
                count = class_counts.get(i, 0)
                if count > 0:
                    print(f"  {class_name}: {count} annotations")
                else:
                    print(f"  {class_name}: 0 annotations")
    
    # Overall statistics
    print(f"\n--- Overall Statistics ---")
    print(f"Total images across all splits: {total_images}")
    
    # Identify class imbalance
    print(f"\n--- Class Imbalance Analysis ---")
    train_counts = all_counts.get("Training", Counter())
    
    if train_counts:
        sorted_counts = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
        print("Most frequent classes:")
        for class_id, count in sorted_counts[:5]:
            print(f"  {class_names[class_id]}: {count} annotations")
            
        print("Least frequent classes:")
        for class_id, count in sorted_counts[-5:]:
            print(f"  {class_names[class_id]}: {count} annotations")
        
        # Calculate imbalance ratio
        max_count = max(train_counts.values()) if train_counts else 1
        min_count = min(train_counts.values()) if train_counts else 1
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 5:
            print("⚠️  WARNING: Significant class imbalance detected!")
        elif imbalance_ratio > 2:
            print("⚠️  NOTICE: Moderate class imbalance detected")
        else:
            print("✅ Class distribution is relatively balanced")
    
    # Check for potential issues
    print(f"\n--- Potential Issues ---")
    issues = []
    
    if train_counts:
        zero_annotation_classes = [i for i in range(len(class_names)) 
                                 if train_counts.get(i, 0) == 0]
        if zero_annotation_classes:
            issues.append(f"Classes with zero annotations: {', '.join([class_names[i] for i in zero_annotation_classes])}")
    
    for split_name, class_counts in all_counts.items():
        very_low_count_classes = [i for i, count in class_counts.items() 
                                if count < 5]
        if very_low_count_classes:
            issues.append(f"{split_name}: Classes with very few annotations (<5): {', '.join([class_names[i] for i in very_low_count_classes])}")
    
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
    else:
        print("✅ No major issues detected")
    
    return all_counts

if __name__ == "__main__":
    data_path = "LATERAL ORTHO AI.v2i.yolov11"
    analyze_dataset(data_path)
