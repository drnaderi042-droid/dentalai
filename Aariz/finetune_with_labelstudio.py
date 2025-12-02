#!/usr/bin/env python3
"""
Fine-tune Aariz model with additional landmarks using LabelStudio annotations
"""
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fine-tune model with additional landmarks')
    parser.add_argument('--annotations_dir', required=True, help='Directory with new annotations')
    parser.add_argument('--checkpoint', default='Aariz/checkpoint_best_512.pth', help='Base model checkpoint')
    parser.add_argument('--landmark_symbols', nargs='+', required=True, help='New landmark symbols')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    # Command to run fine-tuning
    cmd = f"""
python Aariz/finetune_extended_landmarks.py \\
    --dataset_path Aariz/Aariz \\
    --checkpoint {args.checkpoint} \\
    --additional_landmarks {' '.join(args.landmark_symbols)} \\
    --batch_size {args.batch_size} \\
    --epochs {args.epochs} \\
    --lr {args.lr} \\
    --image_size 512 512 \\
    --save_dir checkpoints_extended \\
    --log_dir logs_extended \\
    --model hrnet \\
    --mixed_precision
    """

    print("Running fine-tuning command:")
    print(cmd)

    # Execute command
    os.system(cmd)

if __name__ == "__main__":
    main()
