#!/usr/bin/env python3
"""
LabelStudio Setup for Cephalometric Landmark Annotation
Sets up LabelStudio project for annotating cephalometric landmarks on X-ray images
"""
import os
import json
import argparse
from pathlib import Path

def create_labelstudio_config(num_additional_landmarks=2, landmark_symbols=None):
    """
    Create LabelStudio project configuration for cephalometric landmark annotation
    """
    if landmark_symbols is None:
        landmark_symbols = [f"LM{i+1}" for i in range(num_additional_landmarks)]

    # Base cephalometric landmarks (29 standard landmarks)
    base_landmarks = [
        "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R", "S", "Ar", "Co", "Gn", "Go", "Po",
        "LPM", "LIT", "LMT", "UPM", "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N'", "Pog'", "Sn"
    ]

    # Create label configuration
    labels = []
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080",
        "#808000", "#800080", "#008080", "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0"
    ]

    # Add base landmarks
    for i, landmark in enumerate(base_landmarks):
        color = colors[i % len(colors)]
        labels.append({
            "value": landmark,
            "background": color,
            "showInline": True
        })

    # Add new landmarks with distinct colors
    for i, landmark in enumerate(landmark_symbols):
        color = "#FF4500" if i == 0 else "#FF6347"  # Orange colors for new landmarks
        labels.append({
            "value": landmark,
            "background": color,
            "showInline": True
        })

    # Create LabelStudio configuration
    label_config = '<View>\n  <Image name="image" value="$image"/>\n  <KeyPointLabels name="landmarks" toName="image">\n'
    for label in labels:
        label_config += f'    <Label value="{label["value"]}" background="{label["background"]}"/>\n'
    label_config += '  </KeyPointLabels>\n</View>'

    config = {
        "title": "Cephalometric Landmark Annotation",
        "description": f"Annotation of {len(base_landmarks)} base + {len(landmark_symbols)} new cephalometric landmarks",
        "label_config": label_config,
        "expert_instruction": f"""
Annotate cephalometric landmarks on the X-ray image:

BASE LANDMARKS (29):
- A: A-point (maxilla)
- ANS: Anterior Nasal Spine
- B: B-point (mandible)
- Me: Menton
- N: Nasion
- Or: Orbitale
- Pog: Pogonion
- PNS: Posterior Nasal Spine
- Pn: Pronasale
- R: Ramus
- S: Sella
- Ar: Articulare
- Co: Condylion
- Gn: Gnathion
- Go: Gonion
- Po: Porion
- LPM: Lower 2nd PM Cusp Tip
- LIT: Lower Incisor Tip
- LMT: Lower Molar Cusp Tip
- UPM: Upper 2nd PM Cusp Tip
- UIA: Upper Incisor Apex
- UIT: Upper Incisor Tip
- UMT: Upper Molar Cusp Tip
- LIA: Lower Incisor Apex
- Li: Labrale inferius
- Ls: Labrale superius
- N': Soft Tissue Nasion
- Pog': Soft Tissue Pogonion
- Sn: Subnasale

NEW LANDMARKS TO ANNOTATE:
""" + "\\n".join([f"- {symbol}: [Description needed]" for symbol in landmark_symbols]) + """

INSTRUCTIONS:
1. Click on each landmark point precisely
2. Use zoom and pan to ensure accuracy
3. Annotate all visible landmarks
4. New landmarks are optional - only annotate if clearly visible
        """
    }

    return config

def setup_labelstudio_project(image_dir, output_dir, num_additional_landmarks=2, landmark_symbols=None):
    """
    Set up LabelStudio project files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create project configuration
    config = create_labelstudio_config(num_additional_landmarks, landmark_symbols)

    # Save configuration
    with open(os.path.join(output_dir, 'project_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Create tasks from images
    tasks = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for image_file in os.listdir(image_dir):
        if any(image_file.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_dir, image_file)
            task = {
                "data": {
                    "image": image_path
                },
                "meta": {
                    "image_name": image_file
                }
            }
            tasks.append(task)

    # Save tasks
    with open(os.path.join(output_dir, 'tasks.json'), 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"LabelStudio project setup complete!")
    print(f"Project files saved to: {output_dir}")
    print(f"Number of images found: {len(tasks)}")
    print(f"Total landmarks to annotate: 29 + {num_additional_landmarks}")
    print(f"New landmarks: {landmark_symbols}")

    return config, tasks

def create_annotation_converter(output_dir):
    """
    Create script to convert LabelStudio annotations to Aariz format
    """
    converter_script = '''#!/usr/bin/env python3
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
'''

    with open(os.path.join(output_dir, 'convert_annotations.py'), 'w') as f:
        f.write(converter_script)

    os.chmod(os.path.join(output_dir, 'convert_annotations.py'), 0o755)

def main():
    parser = argparse.ArgumentParser(description='Setup LabelStudio for cephalometric landmark annotation')
    parser.add_argument('--image_dir', required=True, help='Directory containing X-ray images')
    parser.add_argument('--output_dir', default='labelstudio_project', help='Output directory for project files')
    parser.add_argument('--num_landmarks', type=int, default=2, help='Number of additional landmarks')
    parser.add_argument('--landmark_symbols', nargs='+', help='Landmark symbols (e.g., PT PTL)')

    args = parser.parse_args()

    # Default landmark symbols if not provided
    if args.landmark_symbols is None:
        args.landmark_symbols = [f"LM{i+1}" for i in range(args.num_landmarks)]

    print("Setting up LabelStudio project for cephalometric landmark annotation...")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Additional landmarks: {args.landmark_symbols}")

    # Setup project
    config, tasks = setup_labelstudio_project(
        args.image_dir,
        args.output_dir,
        args.num_landmarks,
        args.landmark_symbols
    )

    # Create converter
    create_annotation_converter(args.output_dir)

    print("\\nNext steps:")
    print("1. Install LabelStudio: pip install label-studio")
    print("2. Start LabelStudio: label-studio")
    print("3. Import project from: labelstudio_project/project_config.json")
    print("4. Import tasks from: labelstudio_project/tasks.json")
    print("5. Start annotating images")
    print("6. Export annotations and convert using: python convert_annotations.py")

if __name__ == "__main__":
    main()
