#!/bin/bash
set -e

echo "================================================================================"
echo "🚀 Starting YOLOv8 Training on AMD RX 5600 XT"
echo "================================================================================"
echo ""

# Kill any existing training
pkill -9 -f "train_optimized_v4_fixed.py" 2>/dev/null || true
sleep 2

# Check GPU
echo "📊 GPU Status:"
rocm-smi --showtemp --showuse 2>/dev/null | grep -E "GPU\[0\]" | head -3
echo ""

# Activate venv
cd /home/kevin/Projects/robust-thermal-image-object-detection
source venv-py310-rocm52/bin/activate

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="training_${TIMESTAMP}.log"

echo "📝 Log file: $LOGFILE"
echo "🏃 Starting training..."
echo ""

# Run with unbuffered output and proper logging
stdbuf -oL -eL python -u train_optimized_v4_fixed.py 2>&1 | tee "$LOGFILE"

echo ""
echo "================================================================================"
echo "✅ Training completed or stopped"
echo "================================================================================"
