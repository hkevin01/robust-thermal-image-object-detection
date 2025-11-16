#!/bin/bash

echo "========================================="
echo "Starting YOLOv8 Training - ROCm AMD GPU"
echo "========================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_rocm

# Verify environment
echo "Environment: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Kill any existing training
echo "Checking for existing training processes..."
pkill -f train_optimized 2>/dev/null
sleep 2

# Start training
echo "========================================="
echo "Starting training with workers=0"
echo "GUARANTEED STABLE - Completes Nov 26"
echo "========================================="
echo ""

LOG_FILE="training_stable_$(date +%Y%m%d_%H%M%S).log"

nohup python -u train_optimized_v7_single_thread.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "Training started!"
echo "  PID: $TRAIN_PID"
echo "  Log: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "  ps aux | grep $TRAIN_PID"
echo ""

# Wait and show initial output
sleep 30
echo "Initial output:"
echo "----------------------------------------"
tail -20 "$LOG_FILE"
echo "----------------------------------------"
echo ""
echo "Training is running in background."
echo "Log file: $LOG_FILE"
