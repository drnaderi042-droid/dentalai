"""
Analyze class distribution in training dataset
"""
from pathlib import Path
from collections import Counter

base_dir = Path(__file__).parent
labels_dir = base_dir / 'LATERAL ORTHO AI.v2i.yolov8' / 'train' / 'labels'

# Class names from data.yaml
class_names = ['Class I', 'Class II', 'Class III', 'Cross bite', 'Crowding', 'Deep bite', 'Open bite', 'Proclination', 'Retroclination', 'Rotation', 'Spacing']

# Count occurrences of each class
class_counts = Counter()
total_images = 0
images_with_class_i = 0
images_with_class_ii = 0
images_with_class_iii = 0

label_files = list(labels_dir.glob('*.txt'))
print(f"Analyzing {len(label_files)} label files...\n")

for label_file in label_files:
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if lines:
                total_images += 1
                has_class_i = False
                has_class_ii = False
                has_class_iii = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            class_counts[class_name] += 1
                            
                            if class_name == 'Class I':
                                has_class_i = True
                            elif class_name == 'Class II':
                                has_class_ii = True
                            elif class_name == 'Class III':
                                has_class_iii = True
                
                if has_class_i:
                    images_with_class_i += 1
                if has_class_ii:
                    images_with_class_ii += 1
                if has_class_iii:
                    images_with_class_iii += 1
    except Exception as e:
        print(f"Error reading {label_file.name}: {e}")

print(f"Class Distribution in Training Dataset:")
print(f"   Total images: {total_images}")
print(f"\n   Images containing each class:")
if total_images > 0:
    print(f"      Class I: {images_with_class_i} images ({images_with_class_i/total_images*100:.1f}%)")
    print(f"      Class II: {images_with_class_ii} images ({images_with_class_ii/total_images*100:.1f}%)")
    print(f"      Class III: {images_with_class_iii} images ({images_with_class_iii/total_images*100:.1f}%)")

print(f"\n   Total detections per class:")
for class_name in class_names:
    count = class_counts[class_name]
    total_detections = sum(class_counts.values())
    percentage = (count / total_detections * 100) if total_detections > 0 else 0
    print(f"      {class_name}: {count} detections ({percentage:.1f}%)")

print(f"\n   Top 5 most common classes:")
for class_name, count in class_counts.most_common(5):
    print(f"      {class_name}: {count}")

if total_images > 0 and images_with_class_i < images_with_class_ii and images_with_class_i < images_with_class_iii:
    print(f"\nWARNING: Class I appears in fewer images than Class II/III")
    print(f"   This may explain why the model has difficulty detecting Class I")

