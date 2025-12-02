#!/bin/bash
# Training Script for 1024x1024 Multi-GPU using DistributedDataParallel
# This script uses torchrun for proper DDP setup - BEST for multi-GPU!

echo "========================================"
echo "Training Aariz Model - 1024x1024 Multi-GPU (DDP)"
echo "========================================"
echo ""

# Check GPU availability
python3 -c "import torch; print('GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 'No GPU found')" 2>/dev/null

echo ""
echo "Using DistributedDataParallel (DDP) with torchrun"
echo "This ensures BOTH GPUs are properly used!"
echo ""
echo "Training settings:"
echo "  Image Size: 1024 x 1024"
echo "  Batch Size: 2 per GPU (with 2 GPUs = 4 total) - Safe for 12GB VRAM"
echo "  Gradient Accumulation: 4 steps"
echo "  Effective Batch Size: 16 (2 x 2 GPUs x 4 accumulation)"
echo "  Learning Rate: 3e-4"
echo "  Epochs: 200"
echo "  Expected VRAM: ~7-8GB per GPU"
echo ""
echo "Note: If you want to use more VRAM (~10GB), try:"
echo "  --batch_size 3 --num_workers 4 (may cause OOM)"
echo ""

# Check for checkpoint
if [ -f "checkpoints_1024x1024/checkpoint_best.pth" ]; then
    echo "Checkpoint found. Resuming..."
    RESUME_FLAG="--resume checkpoints_1024x1024/checkpoint_best.pth"
else
    echo "No checkpoint found. Starting new training..."
    RESUME_FLAG=""
fi

# Use torchrun for DDP (this ensures both GPUs are used!)
# Using batch_size=2 for stability (can try 3 if VRAM allows)
torchrun --nproc_per_node=2 train_1024x1024.py \
    $RESUME_FLAG \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 200 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --mixed_precision \
    --use_ema \
    --use_ddp \
    --num_workers 4

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"

