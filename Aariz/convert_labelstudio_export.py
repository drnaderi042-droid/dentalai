#!/usr/bin/env python3
"""
Convert LabelStudio export to Aariz dataset format for the 18 annotated images
"""
import json
import os
from pathlib import Path

def convert_labelstudio_to_aariz(labelstudio_export_file, output_dir):
    """
    Convert LabelStudio export to Aariz JSON format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load LabelStudio export
    with open(labelstudio_export_file, 'r', encoding='utf-8') as f:
        labelstudio_data = json.load(f)

    print(f"Processing {len(labelstudio_data)} annotated images...")

    for task in labelstudio_data:
        try:
            # Get image name from file upload
            image_name = task['file_upload'].split('-', 1)[1].rsplit('.', 1)[0]  # Remove prefix and extension

            # Extract new landmarks from LabelStudio results
            new_landmarks = []
            # LabelStudio export has results in task['annotations'][0]['result']
            if 'annotations' in task and task['annotations']:
                for result in task['annotations'][0].get('result', []):
                    if result['type'] == 'keypointlabels':
                        landmark_symbol = result['value']['keypointlabels'][0]
                        if landmark_symbol in ['p1', 'p2']:
                            # Convert from percentage back to pixels (assuming 512x512 images)
                            x_pixel = int((result['value']['x'] / 100) * 512)
                            y_pixel = int((result['value']['y'] / 100) * 512)

                            new_landmark = {
                                "symbol": landmark_symbol,
                                "value": {
                                    "x": x_pixel,
                                    "y": y_pixel
                                }
                            }
                            new_landmarks.append(new_landmark)

            # Load existing annotations for this image
            existing_annotation_file = f"Aariz/Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists/{image_name}.json"
            existing_landmarks = []

            if os.path.exists(existing_annotation_file):
                with open(existing_annotation_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_landmarks = existing_data['landmarks']
                print(f"Found existing annotations for {image_name}: {len(existing_landmarks)} landmarks")
            else:
                print(f"Warning: No existing annotations found for {image_name}")

            # Combine existing and new landmarks
            all_landmarks = existing_landmarks + new_landmarks

            # Create merged annotation
            merged_data = {
                "ceph_id": image_name,
                "landmarks": all_landmarks,
                "dataset_name": "Extended Aariz Dataset with p1 and p2 landmarks",
                "base_landmarks_count": len(existing_landmarks),
                "new_landmarks_count": len(new_landmarks),
                "total_landmarks": len(all_landmarks)
            }

            # Save merged annotation
            output_file = os.path.join(output_dir, f"{image_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)

            print(f"Converted: {image_name} ({len(existing_landmarks)} + {len(new_landmarks)} = {len(all_landmarks)} landmarks)")

        except Exception as e:
            print(f"Error processing task {task.get('id', 'unknown')}: {e}")
            continue

    print(f"\\nConversion complete! Processed {len(labelstudio_data)} images.")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_labelstudio_export.py <labelstudio_export.json> <output_dir>")
        sys.exit(1)

    labelstudio_file = sys.argv[1]
    output_dir = sys.argv[2]

    convert_labelstudio_to_aariz(labelstudio_file, output_dir)
    print("Conversion complete!")
