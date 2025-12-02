#!/usr/bin/env python3
"""
Convert LabelStudio annotations to Aariz dataset format
"""
import json
import os
from pathlib import Path

def convert_labelstudio_to_aariz(labelstudio_annotations, output_dir):
    """
    Convert LabelStudio export to Aariz JSON format
    """
    os.makedirs(output_dir, exist_ok=True)

    for annotation_file in Path(labelstudio_annotations).glob("*.json"):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract image name
        image_path = data['task']['data']['image']
        image_name = Path(image_path).stem

        # Convert annotations
        landmarks = []
        for result in data['result']:
            if result['type'] == 'keypointlabels':
                for label in result['value']['keypointlabels']:
                    landmark = {
                        "symbol": label['value'],
                        "value": {
                            "x": int(label['x'] * result['original_width'] / 100),
                            "y": int(label['y'] * result['original_height'] / 100)
                        }
                    }
                    landmarks.append(landmark)

        # Create Aariz format
        aariz_annotation = {
            "ceph_id": image_name,
            "landmarks": landmarks,
            "dataset_name": "Extended Aariz Dataset with New Landmarks"
        }

        # Save annotation
        output_file = os.path.join(output_dir, f"{image_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aariz_annotation, f, indent=2, ensure_ascii=False)

        print(f"Converted: {annotation_file.name} -> {image_name}.json")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_annotations.py <labelstudio_output_dir> <aariz_output_dir>")
        sys.exit(1)

    labelstudio_dir = sys.argv[1]
    aariz_dir = sys.argv[2]

    convert_labelstudio_to_aariz(labelstudio_dir, aariz_dir)
    print("Conversion complete!")
