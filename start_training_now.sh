#!/bin/bash
set -e

echo "========================================="
echo "Starting Training with workers=0"
echo "========================================="

# Kill any existing training
pkill -f train_optimized 2>/dev/null || true
sleep 2

# Start training in conda environment
LOG_FILE="training_final_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Log file: $LOG_FILE"

# Use conda run to ensure correct environment
nohup ~/anaconda3/envs/pytorch_rocm/bin/python -u train_optimized_v7_single_thread.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Training started with PID: $PID"
echo "$PID" > training.pid

sleep 15

echo ""
echo "Initial output:"
echo "----------------------------------------"
tail -30 "$LOG_FILE"
echo "----------------------------------------"

echo ""
echo "Monitor with: tail -f $LOG_FILE"
