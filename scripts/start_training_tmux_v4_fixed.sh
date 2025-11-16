#!/bin/bash

# Kill any existing training sessions
tmux kill-session -t yolo_training 2>/dev/null
tmux kill-session -t yolo_training_v2 2>/dev/null
tmux kill-session -t yolo_training_v3 2>/dev/null
tmux kill-session -t yolo_training_v4 2>/dev/null
tmux kill-session -t yolo_training_v4_fixed 2>/dev/null
echo "✓ Killed old sessions"

# Check GPU status
echo ""
echo "GPU Status:"
rocm-smi --showtemp --showuse --showmeminfo vram | grep -E "GPU|Temperature|use|VRAM"
echo ""

# Start new training session
cd /home/kevin/Projects/robust-thermal-image-object-detection
tmux new-session -d -s yolo_training_v4_fixed
tmux send-keys -t yolo_training_v4_fixed "cd /home/kevin/Projects/robust-thermal-image-object-detection" C-m
tmux send-keys -t yolo_training_v4_fixed "source venv-py310-rocm52/bin/activate" C-m
tmux send-keys -t yolo_training_v4_fixed "python train_optimized_v4_fixed.py 2>&1 | tee training_v4_fixed_batch8.log" C-m

echo "✓ Training started in tmux session 'yolo_training_v4_fixed'"
echo "  Batch size: 8 (reduced to avoid OOM)"
echo "  Log file: training_v4_fixed_batch8.log"
echo ""
echo "To monitor:"
echo "  tmux attach -t yolo_training_v4_fixed"
echo "  tail -f training_v4_fixed_batch8.log"
echo "  watch -n 1 rocm-smi"
