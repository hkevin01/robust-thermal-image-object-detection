#!/bin/bash
echo "================================================================================
"
echo "🚀 YOLO TRAINING STATUS - AMD RX 5600 XT"
echo "================================================================================"
echo ""

# Check if training process is running
if pgrep -f "train_optimized_v4_fixed.py" > /dev/null; then
    echo "✅ Training process: RUNNING (PID: $(pgrep -f train_optimized_v4_fixed.py))"
else
    echo "❌ Training process: NOT RUNNING"
fi

echo ""
echo "📊 GPU Status:"
rocm-smi --showtemp --showuse --showmeminfo vram | grep -E "GPU|Temperature|use|VRAM" | head -5

echo ""
echo "📈 Latest Training Progress:"
tail -5 direct_run.log | grep -E "Epoch|box_loss" | tail -1

echo ""
echo "⏱️  Performance Metrics:"
# Extract speed from log
SPEED=$(tail -20 direct_run.log | grep "it/s" | tail -1 | grep -oP '\d+\.\d+it/s' | head -1)
echo "  Current speed: $SPEED"

echo ""
echo "📁 Training Output:"
ls -lh runs/detect/ | tail -3

echo ""
echo "================================================================================
"
