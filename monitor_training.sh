#!/bin/bash

echo "==== YOLOv8 Training Monitor ===="
echo ""

# Check if training is running
echo "📊 Training Status:"
if tmux has-session -t yolo_training_fixed 2>/dev/null; then
    echo "  ✅ tmux session 'yolo_training_fixed' is ACTIVE"
    
    # Get process info
    PROC_COUNT=$(ps aux | grep train_optimized_v4_fixed.py | grep -v grep | wc -l)
    echo "  ✅ $PROC_COUNT training process(es) running"
    
    # Show current training line
    echo ""
    echo "📈 Current Progress:"
    tmux capture-pane -t yolo_training_fixed -p | tail -3 | head -2
    echo ""
    
else
    echo "  ❌ No training session found"
    exit 1
fi

# GPU Status
echo "🖥️  GPU Status:"
rocm-smi --showtemp --showuse --showmeminfo vram 2>/dev/null | grep -E "Temperature|use \(%\)|Used Memory" | sed 's/^/  /'
echo ""

# Check latest model weights
echo "💾 Saved Checkpoints:"
LATEST_RUN=$(ls -dt runs/detect/train_optimized_v4_fixed* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    echo "  📁 Latest run: $LATEST_RUN"
    if [ -d "$LATEST_RUN/weights" ]; then
        WEIGHTS=$(ls -lht "$LATEST_RUN/weights/"*.pt 2>/dev/null | head -3)
        if [ -n "$WEIGHTS" ]; then
            echo "$WEIGHTS" | awk '{print "    " $9 " (" $5 ", " $6 " " $7 ")"}'
        else
            echo "    ⏳ No checkpoints saved yet"
        fi
    fi
else
    echo "  ⏳ No runs found yet"
fi

echo ""
echo "Commands:"
echo "  • View live: tmux attach -t yolo_training_fixed"
echo "  • Detach: Press Ctrl+B then D"
echo "  • Stop: tmux kill-session -t yolo_training_fixed"
echo "  • Re-run: $0"
