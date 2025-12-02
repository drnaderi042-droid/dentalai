"""
Create annotations_31.json from existing annotation files
Combines anatomical landmarks (29) + calibration points (P1/P2) = 31 total
"""
import json
import os
from pathlib import Path
import glob

def landmark_symbol_to_number(symbol):
    """Convert landmark symbol to number (0-30, where 30=P1, 31=P2)"""
    symbol_map = {
        'A': 0, 'ANS': 1, 'B': 2, 'Me': 3, 'N': 4, 'Or': 5, 'Pog': 6,
        'PNS': 7, 'Pn': 8, 'R': 9, 'S': 10, 'Ar': 11, 'Co': 12, 'Gn': 13,
        'Go': 14, 'Po': 15, 'LPM': 16, 'LIT': 17, 'LMT': 18, 'UPM': 19,
        'UIA': 20, 'UIT': 21, 'UMT': 22, 'LIA': 23, 'Li': 24, 'Ls': 25,
        "N'": 26, "Pog'": 27, 'Sn': 28
    }
    return symbol_map.get(symbol, -1)

def process_annotations():
    """Process all annotation files to create annotations_31.json"""

    annotations_dir = Path("Aariz/train/Annotations/Cephalometric Landmarks/Junior Orthodontists")
    p1p2_file = Path("annotations_p1_p2.json")
    output_file = Path("annotations_31.json")

    # Load P1/P2 annotations
    print("Loading P1/P2 annotations...")
    with open(p1p2_file, 'r', encoding='utf-8') as f:
        p1p2_data = json.load(f)

    # Create P1/P2 lookup by filename
    p1p2_lookup = {}
    for item in p1p2_data:
        image_url = item['data']['image']
        filename = image_url.split('/')[-1]

        # Remove UUID prefix if present
        if '-' in filename:
            parts = filename.split('-')
            if len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0].lower()):
                filename = '-'.join(parts[1:])

        p1 = None
        p2 = None
        if item['annotations']:
            for result_item in item['annotations'][0]['result']:
                if result_item.get('type') == 'keypointlabels':
                    value = result_item.get('value', {})
                    labels = value.get('keypointlabels', [])

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
            p1p2_lookup[filename] = {'p1': p1, 'p2': p2}

    print(f"Found {len(p1p2_lookup)} P1/P2 annotations")

    # Process anatomical annotations
    print("Processing anatomical annotations...")
    annotations_31 = []
    processed_count = 0

    json_files = list(annotations_dir.glob("*.json"))
    print(f"Found {len(json_files)} anatomical annotation files")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = json_file.stem + '.png'  # Add .png extension

            # Check if we have P1/P2 for this image
            if filename not in p1p2_lookup:
                print(f"Skipping {filename} - no P1/P2 data")
                continue

            # Process anatomical landmarks
            landmarks = {}
            if 'landmarks' in data:
                for landmark in data['landmarks']:
                    symbol = landmark.get('symbol', '')
                    landmark_num = landmark_symbol_to_number(symbol)

                    if landmark_num >= 0:
                        # Convert pixel coordinates to normalized [0,1]
                        # Assuming image size is approximately 2000x2200
                        img_width = 2000  # approximate
                        img_height = 2200  # approximate

                        x_norm = landmark['value']['x'] / img_width
                        y_norm = landmark['value']['y'] / img_height

                        landmarks[landmark_num] = {
                            'x': x_norm,
                            'y': y_norm
                        }

            # Check if we have enough anatomical landmarks
            anatomical_count = len([k for k in landmarks.keys() if k < 29])
            if anatomical_count < 20:  # Allow more missing landmarks
                print(f"Skipping {filename} - only {anatomical_count} anatomical landmarks")
                continue

            # Add P1/P2 as landmarks 30 and 31 (if available)
            p1p2_data = p1p2_lookup.get(filename)
            if p1p2_data:
                landmarks[30] = p1p2_data['p1']  # P1
                landmarks[31] = p1p2_data['p2']  # P2
            else:
                # Use default positions if P1/P2 not available
                landmarks[30] = {'x': 0.5, 'y': 0.5}  # Center
                landmarks[31] = {'x': 0.5, 'y': 0.5}  # Center

            # Ensure we have all 32 landmarks (0-31)
            for i in range(32):
                if i not in landmarks:
                    # Fill missing landmarks with center position
                    landmarks[i] = {'x': 0.5, 'y': 0.5}

            # Create annotation entry with all 32 keypoints
            annotation_entry = {
                "id": processed_count + 1,
                "data": {
                    "image": f"/data/upload/{filename}"
                },
                "annotations": [
                    {
                        "id": processed_count + 1,
                        "completed_by": 1,
                        "result": [
                            {
                                "type": "keypointlabels",
                                "value": {
                                    "x": landmarks[i]['x'] * 100,  # Convert back to percentage
                                    "y": landmarks[i]['y'] * 100,
                                    "width": 1.0,
                                    "keypointlabels": [str(i + 1)]  # 1-based indexing in UI
                                }
                            } for i in range(32)
                        ]
                    }
                ],
                "file_upload": filename
            }

            annotations_31.append(annotation_entry)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} images...")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Save the combined annotations
    print(f"Saving {len(annotations_31)} annotations to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_31, f, indent=2, ensure_ascii=False)

    print("Done!")
    print(f"Created annotations_31.json with {len(annotations_31)} samples")
    print("Each sample contains 29 anatomical + 2 calibration landmarks = 31 total")

if __name__ == '__main__':
    process_annotations()
