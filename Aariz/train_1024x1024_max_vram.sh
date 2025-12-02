#!/bin/bash
# Training Script for 1024x1024 Multi-GPU - Maximum VRAM Usage (~10GB per GPU)
# Uses gradient accumulation instead of larger batch size to avoid OOM

echo "========================================"
echo "Training Aariz Model - 1024x1024 Multi-GPU (Max VRAM)"
echo "========================================"
echo ""

# Check GPU availability
python3 -c "import torch; print('GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 'No GPU found')" 2>/dev/null

echo ""
echo "Strategy: Use batch_size=2 + higher gradient accumulation"
echo "This maximizes VRAM usage (~10GB) without OOM risk"
echo ""
echo "Training settings:"
echo "  Image Size: 1024 x 1024"
echo "  Batch Size: 2 per GPU (safe, avoids OOM)"
echo "  Gradient Accumulation: 6 steps (increased for larger effective batch)"
echo "  Effective Batch Size: 24 (2 x 2 GPUs x 6 accumulation)"
echo "  Expected VRAM: ~9-10GB per GPU"
echo "  Learning Rate: 3e-4"
echo "  Epochs: 200"
echo ""

# Check for checkpoint
if [ -f "checkpoints_1024x1024/checkpoint_best.pth" ]; then
    echo "Checkpoint found. Resuming..."
    RESUME_FLAG="--resume checkpoints_1024x1024/checkpoint_best.pth"
else
    echo "No checkpoint found. Starting new training..."
    RESUME_FLAG=""
fi

# Use torchrun for DDP with increased gradient accumulation
torchrun --nproc_per_node=2 train_1024x1024.py \
    $RESUME_FLAG \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 6 \
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

















