#!/bin/bash

# Check if GPU is available
echo "Checking GPU status..."
rocm-smi --showuse --showtemp 2>&1 | head -20

# Session name
SESSION_NAME="yolo_training"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo ""
    echo "⚠️  Training session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && ./$0"
    echo ""
    exit 1
fi

# Create log filename with timestamp
LOG_FILE="logs/training_optimized_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Create new tmux session and run training
echo "Starting training in tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" \
    "python train_optimized_v2.py 2>&1 | tee '$LOG_FILE'"

echo ""
echo "✅ Training started successfully in tmux session '$SESSION_NAME'"
echo ""
echo "📊 Monitoring Commands:"
echo "  • Attach to session:  tmux attach -t $SESSION_NAME"
echo "  • Detach from session: Ctrl+B then D"
echo "  • List sessions:      tmux ls"
echo "  • Kill session:       tmux kill-session -t $SESSION_NAME"
echo ""
echo "📝 Log file: $LOG_FILE"
echo ""
echo "💡 Quick commands:"
echo "  • Watch log live:     tail -f $LOG_FILE"
echo "  • Check GPU:          rocm-smi"
echo "  • Check progress:     grep 'Epoch' $LOG_FILE | tail -5"
echo ""
