#!/bin/bash
# Start standalone training in screen session
# This runs independently of VS Code

SESSION_NAME="yolo_training"

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "⚠️ Training session '$SESSION_NAME' already exists"
    echo "To attach: screen -r $SESSION_NAME"
    echo "To kill: screen -X -S $SESSION_NAME quit"
    exit 1
fi

echo "🚀 Starting training in screen session: $SESSION_NAME"
echo "📝 To attach: screen -r $SESSION_NAME"
echo "📝 To detach: Ctrl+A then D"
echo "📝 To check status: screen -ls"
echo ""

# Start training in detached screen
screen -dmS "$SESSION_NAME" bash -c "
    cd ~/Projects/robust-thermal-image-object-detection
    source venv-py310-rocm52/bin/activate
    python train_standalone.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
"

sleep 2

if screen -list | grep -q "$SESSION_NAME"; then
    echo "✅ Training started successfully!"
    echo "   Session: $SESSION_NAME"
    echo "   Log: logs/training_$(date +%Y%m%d)_*.log"
    echo ""
    echo "To monitor: screen -r $SESSION_NAME"
else
    echo "❌ Failed to start training session"
    exit 1
fi
