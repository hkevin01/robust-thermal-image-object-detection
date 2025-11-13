#!/bin/bash
#
# Start Optimized YOLOv8 Training in Screen Session
# ==================================================
#
# Uses optimized Conv2d fallback (im2col + rocBLAS)
# Expected: ~18 batches/sec, 62 hours for 50 epochs
#
# Hardware: AMD RX 5600 XT (gfx1010)
#

set -e

PROJECT_DIR="$HOME/Projects/robust-thermal-image-object-detection"
VENV_PATH="$PROJECT_DIR/venv-py310-rocm52"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_optimized_${TIMESTAMP}.log"
SCREEN_NAME="yolo_optimized"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Optimized YOLOv8 Training - AMD RX 5600 XT (gfx1010)        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • Project: $PROJECT_DIR"
echo "  • Virtual env: $VENV_PATH"
echo "  • Log file: $LOG_FILE"
echo "  • Screen session: $SCREEN_NAME"
echo ""

# Check if screen session already exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "⚠️  WARNING: Screen session '$SCREEN_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: screen -r $SCREEN_NAME"
    echo "  2. Kill existing session first: screen -X -S $SCREEN_NAME quit"
    echo ""
    read -p "Kill existing session and start new training? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing existing session..."
        screen -X -S $SCREEN_NAME quit
        sleep 2
    else
        echo "Aborted. Attach with: screen -r $SCREEN_NAME"
        exit 0
    fi
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Check GPU status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rocm-smi --showuse --showtemp || echo "Could not query GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create screen session with training command
echo "Starting training in screen session '$SCREEN_NAME'..."
echo ""

screen -dmS "$SCREEN_NAME" bash -c "
    cd '$PROJECT_DIR' && \
    source '$VENV_PATH/bin/activate' && \
    python train_optimized.py 2>&1 | tee '$LOG_FILE'
"

# Wait a moment for screen to start
sleep 2

# Check if screen session started successfully
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "✓ Training started successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Management Commands:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  • Attach to training:  screen -r $SCREEN_NAME"
    echo "  • Detach from screen:  Ctrl+A then D"
    echo "  • View log:            tail -f $LOG_FILE"
    echo "  • Stop training:       screen -X -S $SCREEN_NAME quit"
    echo "  • List sessions:       screen -ls"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Training Timeline:"
    echo "  • Expected duration: 62.6 hours (2.6 days)"
    echo "  • Start time: $(date)"
    echo "  • Estimated completion: $(date -d '+63 hours')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Monitoring:"
    echo "  • Check status: ./check_training_status.sh"
    echo "  • Live view: screen -r $SCREEN_NAME (Ctrl+A D to detach)"
    echo ""
else
    echo "✗ Failed to start screen session!"
    echo ""
    echo "Check for errors:"
    echo "  • Log file: $LOG_FILE"
    echo "  • Screen sessions: screen -ls"
    exit 1
fi
