#!/usr/bin/env python3
"""
Create LabelStudio project for adding new landmarks to existing annotated images
Shows existing 29 landmarks as read-only and allows adding only new landmarks (p1, p2)
"""
import os
import json
import argparse
from pathlib import Path
import random

def convert_existing_annotations_to_labelstudio(image_dir, annotation_dir, output_dir, num_samples=100):
    """
    Convert existing Aariz annotations to LabelStudio format with pre-annotations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of annotation files
    annotation_files = list(Path(annotation_dir).glob("*.json"))
    print(f"Found {len(annotation_files)} annotation files")

    # Randomly select samples
    if num_samples < len(annotation_files):
        annotation_files = random.sample(annotation_files, num_samples)
        print(f"Selected {num_samples} random samples")

    tasks = []
    processed_count = 0

    for annotation_file in annotation_files:
        try:
            # Load annotation
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_name = data['ceph_id']
            image_path = os.path.join(image_dir, f"{image_name}.png")
            image_path_jpg = os.path.join(image_dir, f"{image_name}.jpg")

            # Check if image exists
            if os.path.exists(image_path):
                actual_image_path = image_path
            elif os.path.exists(image_path_jpg):
                actual_image_path = image_path_jpg
            else:
                print(f"Warning: Image not found for {image_name}, skipping...")
                continue

            # Convert landmarks to LabelStudio format
            predictions = []
            for landmark in data['landmarks']:
                # Convert coordinates (assuming 512x512 images, but LabelStudio uses percentages)
                x_percent = (landmark['value']['x'] / 512) * 100
                y_percent = (landmark['value']['y'] / 512) * 100

                prediction = {
                    "type": "keypointlabels",
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": 0.5,  # Small width for point display
                        "height": 0.5,
                        "rotation": 0,
                        "keypointlabels": [landmark['symbol']]
                    },
                    "to_name": "image",
                    "from_name": "landmarks",
                    "origin": "manual"  # Mark as existing annotation
                }
                predictions.append(prediction)

            task = {
                "data": {
                    "image": actual_image_path
                },
                "meta": {
                    "image_name": image_name
                },
                "predictions": [{
                    "model_version": "existing_annotations",
                    "result": predictions
                }]
            }

            tasks.append(task)
            processed_count += 1

            if processed_count % 10 == 0:
                print(f"Processed {processed_count} images...")

        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            continue

    # Save tasks
    with open(os.path.join(output_dir, 'tasks_with_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Successfully created LabelStudio tasks for {len(tasks)} images")
    return tasks

def create_incremental_config(output_dir):
    """
    Create LabelStudio config for incremental annotation (only new landmarks)
    """
    # Only include the new landmarks p1 and p2
    label_config = '''<View>
  <Image name="image" value="$image"/>
  <KeyPointLabels name="landmarks" toName="image">
    <Label value="p1" background="#FF4500"/>
    <Label value="p2" background="#FF6347"/>
  </KeyPointLabels>
</View>'''

    config = {
        "title": "Add New Landmarks (p1, p2) to Existing Annotations",
        "description": "Add only p1 and p2 landmarks to images with existing 29 base landmarks",
        "label_config": label_config,
        "expert_instruction": """Add new landmarks p1 and p2 to images that already have 29 base landmarks annotated.

EXISTING LANDMARKS (29 base landmarks are already annotated and visible):
- A, ANS, B, Me, N, Or, Pog, PNS, Pn, R, S, Ar, Co, Gn, Go, Po
- LPM, LIT, LMT, UPM, UIA, UIT, UMT, LIA, Li, Ls, N', Pog', Sn

NEW LANDMARKS TO ADD:
- p1: [Description needed] - Orange color
- p2: [Description needed] - Light orange color

INSTRUCTIONS:
1. You will see existing landmarks already marked on the image
2. Only add p1 and p2 landmarks where clearly visible
3. Use zoom and pan for better accuracy
4. Click precisely on the anatomical points
5. If p1 or p2 is not visible/clear, you can skip it

The existing 29 landmarks are pre-annotated and should not be modified."""
    }

    with open(os.path.join(output_dir, 'incremental_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Created incremental config: {os.path.join(output_dir, 'incremental_config.json')}")

def create_merge_script(output_dir):
    """
    Create script to merge new annotations with existing ones
    """
    merge_script = '''#!/usr/bin/env python3
"""
Merge new landmark annotations (p1, p2) with existing 29 base landmarks
"""
import json
import os
from pathlib import Path

def merge_annotations(existing_annotations_dir, new_labelstudio_export, output_dir):
    """
    Merge LabelStudio export with existing annotations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load new annotations from LabelStudio export
    with open(new_labelstudio_export, 'r', encoding='utf-8') as f:
        labelstudio_data = json.load(f)

    merged_count = 0

    for task in labelstudio_data:
        try:
            # Get image name
            image_path = task['data']['image']
            image_name = Path(image_path).stem

            # Find existing annotation
            existing_annotation_path = os.path.join(existing_annotations_dir, f"{image_name}.json")

            if not os.path.exists(existing_annotation_path):
                print(f"Warning: Existing annotation not found for {image_name}")
                continue

            # Load existing annotation
            with open(existing_annotation_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # Extract new landmarks from LabelStudio results
            new_landmarks = []
            for result in task.get('result', []):
                if result['type'] == 'keypointlabels':
                    for label in result['value']['keypointlabels']:
                        if label in ['p1', 'p2']:  # Only process new landmarks
                            # Convert from percentage back to pixels (assuming 512x512)
                            x_pixel = int((result['value']['x'] / 100) * 512)
                            y_pixel = int((result['value']['y'] / 100) * 512)

                            new_landmark = {
                                "symbol": label,
                                "value": {
                                    "x": x_pixel,
                                    "y": y_pixel
                                }
                            }
                            new_landmarks.append(new_landmark)

            # Merge landmarks
            merged_landmarks = existing_data['landmarks'] + new_landmarks

            # Create merged annotation
            merged_data = {
                "ceph_id": image_name,
                "landmarks": merged_landmarks,
                "dataset_name": "Extended Aariz Dataset with p1 and p2 landmarks",
                "base_landmarks_count": len(existing_data['landmarks']),
                "new_landmarks_count": len(new_landmarks)
            }

            # Save merged annotation
            output_file = os.path.join(output_dir, f"{image_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)

            merged_count += 1
            print(f"Merged: {image_name} ({len(existing_data['landmarks'])} + {len(new_landmarks)} landmarks)")

        except Exception as e:
            print(f"Error processing {task}: {e}")
            continue

    print(f"Successfully merged annotations for {merged_count} images")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python merge_annotations.py <existing_annotations_dir> <labelstudio_export.json> <output_dir>")
        sys.exit(1)

    existing_dir = sys.argv[1]
    labelstudio_export = sys.argv[2]
    output_dir = sys.argv[3]

    merge_annotations(existing_dir, labelstudio_export, output_dir)
    print("Merge complete!")
'''

    with open(os.path.join(output_dir, 'merge_annotations.py'), 'w') as f:
        f.write(merge_script)

    os.chmod(os.path.join(output_dir, 'merge_annotations.py'), 0o755)
    print(f"Created merge script: {os.path.join(output_dir, 'merge_annotations.py')}")

def main():
    parser = argparse.ArgumentParser(description='Create incremental annotation project for adding new landmarks')
    parser.add_argument('--image_dir', required=True, help='Directory containing X-ray images')
    parser.add_argument('--annotation_dir', required=True, help='Directory containing existing JSON annotations')
    parser.add_argument('--output_dir', default='incremental_annotation_project', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to annotate')

    args = parser.parse_args()

    print("Creating incremental annotation project...")
    print(f"Image directory: {args.image_dir}")
    print(f"Annotation directory: {args.annotation_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert existing annotations to LabelStudio format
    print("\\nConverting existing annotations to LabelStudio format...")
    tasks = convert_existing_annotations_to_labelstudio(
        args.image_dir,
        args.annotation_dir,
        args.output_dir,
        args.num_samples
    )

    # Create incremental config (only for new landmarks)
    print("\\nCreating incremental configuration...")
    create_incremental_config(args.output_dir)

    # Create merge script
    print("\\nCreating merge script...")
    create_merge_script(args.output_dir)

    print(f"\\n=== Incremental Annotation Project Created ===")
    print(f"Project files saved to: {args.output_dir}")
    print(f"\\nFiles created:")
    print(f"- incremental_config.json (LabelStudio project config)")
    print(f"- tasks_with_predictions.json (Tasks with existing annotations)")
    print(f"- merge_annotations.py (Script to merge new annotations)")
    print(f"\\nNext steps:")
    print(f"1. pip install label-studio")
    print(f"2. label-studio start")
    print(f"3. Create project and import incremental_config.json")
    print(f"4. Import tasks_with_predictions.json")
    print(f"5. Annotate only p1 and p2 landmarks")
    print(f"6. Export results and run merge script")

if __name__ == "__main__":
    main()
