import json

with open('Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists/project-6-at-2025-11-11-03-58-db627e7c.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

count_p1 = 0
count_p2 = 0
count_both = 0

for idx, item in enumerate(data):
    if 'annotations' in item and item['annotations']:
        ann = item['annotations'][0]
        result = ann.get('result', [])
        
        p1_found = False
        p2_found = False
        
        for r in result:
            if r.get('type') == 'keypointlabels':
                labels = r.get('value', {}).get('keypointlabels', [])
                if 'p1' in labels:
                    p1_found = True
                    print(f"Image {idx}: Found p1 at x={r['value']['x']:.1f}, y={r['value']['y']:.1f}")
                if 'p2' in labels:
                    p2_found = True
                    print(f"Image {idx}: Found p2 at x={r['value']['x']:.1f}, y={r['value']['y']:.1f}")
        
        count_p1 += p1_found
        count_p2 += p2_found
        count_both += (p1_found and p2_found)
        
        if not (p1_found and p2_found):
            print(f"Image {idx}: Missing {'p1' if not p1_found else 'p2'}")

print(f"\n=== Summary ===")
print(f"Total images: {len(data)}")
print(f"Images with p1: {count_p1}")
print(f"Images with p2: {count_p2}")
print(f"Images with BOTH p1 and p2: {count_both}")

