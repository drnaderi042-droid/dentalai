import json
from pathlib import Path

# Load annotations
annotations_file = 'Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists/project-6-at-2025-11-11-03-58-db627e7c.json'

with open(annotations_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total items: {len(data)}")
print(f"\n{'='*60}")
print("Checking for p1/p2 annotations...")
print(f"{'='*60}\n")

found_count = 0

for idx, item in enumerate(data):
    if 'annotations' in item and item['annotations']:
        annotation = item['annotations'][0]
        
        if 'result' in annotation:
            p1_found = False
            p2_found = False
            
            for result_item in annotation['result']:
                if result_item.get('type') == 'keypointlabels':
                    value = result_item.get('value', {})
                    labels = value.get('keypointlabels', [])
                    
                    if 'p1' in labels:
                        p1_found = True
                        p1_x = value.get('x', 0)
                        p1_y = value.get('y', 0)
                    elif 'p2' in labels:
                        p2_found = True
                        p2_x = value.get('x', 0)
                        p2_y = value.get('y', 0)
            
            if p1_found and p2_found:
                found_count += 1
                image_filename = item['data'].get('image', '').split('/')[-1]
                print(f"[{found_count}] {image_filename}")
                print(f"    p1: ({p1_x:.2f}, {p1_y:.2f})")
                print(f"    p2: ({p2_x:.2f}, {p2_y:.2f})")
            elif p1_found or p2_found:
                image_filename = item['data'].get('image', '').split('/')[-1]
                print(f"[INCOMPLETE] {image_filename} - Only {'p1' if p1_found else 'p2'} found")

print(f"\n{'='*60}")
print(f"Summary: Found {found_count} images with both p1 and p2")
print(f"{'='*60}")

if found_count == 0:
    print("\nDEBUG: Checking first item structure...")
    print("\nFirst item keys:", list(data[0].keys()))
    print("\nFirst annotation keys:", list(data[0]['annotations'][0].keys()))
    print("\nFirst result item:")
    if data[0]['annotations'][0]['result']:
        result = data[0]['annotations'][0]['result'][0]
        print("  Type:", result.get('type'))
        print("  Value keys:", list(result.get('value', {}).keys()))
        if 'keypointlabels' in result.get('value', {}):
            print("  Keypoint labels:", result['value']['keypointlabels'])
        print("\nFull result item:")
        import pprint
        pprint.pprint(result, depth=3)













