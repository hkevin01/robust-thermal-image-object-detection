#!/bin/bash
# Start YOLOv8 training in tmux - Version 4 (FIXED CLASS MISMATCH)

SESSION_NAME="yolo_training_v4"
LOG_FILE="training_v4.log"

echo "==========================================================================="
echo "AMD RX 5600 XT - Starting Training v4 (Fixed class count)"
echo "==========================================================================="

# Check GPU
echo "GPU Status:"
rocm-smi --showtemp --showmeminfo vram --showuse

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the session
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "source venv-py310-rocm52/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python train_optimized_v4.py 2>&1 | tee $LOG_FILE" C-m

echo "✓ Training started in tmux session '$SESSION_NAME'"
echo ""
echo "Commands:"
echo "  - Attach:  tmux attach -t $SESSION_NAME"
echo "  - Monitor: tail -f $LOG_FILE"
echo "  - GPU:     watch -n 1 rocm-smi"
echo ""
echo "Detach from tmux: Ctrl+B, then D"
echo "==========================================================================="
