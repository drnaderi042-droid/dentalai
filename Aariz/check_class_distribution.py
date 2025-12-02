import os
import json
from collections import Counter

def check_distribution(dataset_path, split):
    """Check class distribution for train/valid/test splits"""
    cvm_folder = os.path.join(dataset_path, split, "Annotations", "CVM Stages")

    if not os.path.exists(cvm_folder):
        print(f"Folder not found: {cvm_folder}")
        return

    stages = []
    for filename in os.listdir(cvm_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(cvm_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'cvm_stage' in data and 'value' in data['cvm_stage']:
                        stage = data['cvm_stage']['value']
                        stages.append(stage)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"\n{split.upper()} SET DISTRIBUTION:")
    print(f"Total samples: {len(stages)}")
    counter = Counter(stages)
    for stage in sorted(counter.keys()):
        count = counter[stage]
        percentage = (count / len(stages)) * 100
        print(f"Stage {stage}: {count} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    dataset_path = "Aariz"

    for split in ["train", "valid", "test"]:
        check_distribution(dataset_path, split)