#!/bin/bash
# Training Script for 1024x1024 Multi-GPU - WSL2/Ubuntu 22.04
# This script is optimized for Linux/WSL2 environment

echo "========================================"
echo "Training Aariz Model - 1024x1024 Multi-GPU (WSL2)"
echo "========================================"
echo ""

# Check GPU availability
python3 -c "import torch; print('GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 'No GPU found')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Could not check GPU status"
fi

echo ""
echo "Training settings:"
echo "  Image Size: 1024 x 1024"
echo "  Batch Size: 2 per GPU (with 2 GPUs = 4 total) - Safe for 12GB VRAM"
echo "  Gradient Accumulation: 4 steps"
echo "  Effective Batch Size: 16 (2 per GPU x 2 GPUs x 4 accumulation)"
echo "  Expected VRAM: ~7-8GB per GPU"
echo "  Learning Rate: 3e-4 (optimized for 1024x1024)"
echo "  Epochs: 200"
echo "  Loss: Adaptive Wing"
echo "  Mixed Precision: Enabled (FP16)"
echo "  EMA: Enabled (for better accuracy)"
echo "  Multi-GPU: Enabled (using both RTX 3060 GPUs)"
echo "  Checkpoints: checkpoints_1024x1024/"
echo "  Logs: logs_1024x1024/"
echo ""

# Check for existing checkpoint
if [ -f "checkpoints_1024x1024/checkpoint_best.pth" ]; then
    echo "Checkpoint found in checkpoints_1024x1024/"
    echo ""
    read -p "Do you want to resume from previous checkpoint? (y/n): " choice
    
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        echo ""
        echo "========================================"
        echo "Resuming training from checkpoint"
        echo "========================================"
        echo ""
        python3 train_1024x1024.py --resume checkpoints_1024x1024/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu --num_workers 4 --use_ddp
    else
        echo ""
        echo "========================================"
        echo "Starting new training"
        echo "========================================"
        echo ""
        python3 train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu --num_workers 4 --use_ddp
    fi
else
    echo "Checkpoint not found - starting new training"
    echo ""
    echo "========================================"
    echo "Starting new training - 1024x1024 Multi-GPU"
    echo "========================================"
    echo ""
    echo "Estimated time: 24-36 hours (with 2x RTX 3060)"
    echo ""
    read -p "Press Enter to continue..."
    echo ""
    echo "Starting training..."
    echo ""
    python3 train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 4 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu
fi

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Results saved in checkpoints_1024x1024/ folder"
echo ""

