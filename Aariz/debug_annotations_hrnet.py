"""Debug script to check annotation parsing"""
import json
from pathlib import Path

annotations_file = 'Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists/project-6-at-2025-11-11-03-58-db627e7c.json'
images_dir = Path('Aariz/train/Cephalograms')

with open(annotations_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total items in JSON: {len(data)}")
print("\nProcessing samples...")

samples_found = 0
for idx, item in enumerate(data):  # Check all items
    print(f"\n--- Item {idx+1} ---")
    
    if 'annotations' in item and item['annotations']:
        annotation = item['annotations'][0]
        
        if 'result' in annotation:
            p1 = None
            p2 = None
            
            print(f"  Result items: {len(annotation['result'])}")
            
            for result_item in annotation['result']:
                if result_item.get('type') == 'keypointlabels':
                    value = result_item.get('value', {})
                    labels = value.get('keypointlabels', [])
                    
                    print(f"    Found keypoint: {labels}, x={value.get('x')}, y={value.get('y')}")
                    
                    if 'p1' in labels:
                        p1 = {
                            'x': value.get('x', 0) / 100.0,
                            'y': value.get('y', 0) / 100.0
                        }
                    elif 'p2' in labels:
                        p2 = {
                            'x': value.get('x', 0) / 100.0,
                            'y': value.get('y', 0) / 100.0
                        }
            
            if p1 and p2:
                # Get image filename
                image_url = item['data'].get('image', '')
                print(f"  data.image: {image_url}")
                
                image_filename = image_url.split('/')[-1]
                print(f"  After split: {image_filename}")
                
                # Remove prefix if exists
                if '-' in image_filename:
                    image_filename = image_filename.split('-', 1)[-1]
                    print(f"  After removing prefix: {image_filename}")
                
                image_path = images_dir / image_filename
                print(f"  Looking for: {image_path}")
                print(f"  Exists: {image_path.exists()}")
                
                if image_path.exists():
                    samples_found += 1
                    print(f"  [OK] Sample {samples_found} found! p1={p1}, p2={p2}")
                else:
                    print(f"  [ERROR] Image file not found!")
            else:
                print(f"  [WARNING] Missing p1 or p2 (p1={bool(p1)}, p2={bool(p2)})")

print(f"\n\nTotal valid samples found: {samples_found}")

