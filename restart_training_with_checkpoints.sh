#!/bin/bash
# Safely stop current training and restart with checkpointing

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Training Restart with Checkpointing                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check current training
TRAIN_PROCS=$(ps aux | grep "python train" | grep -v grep | wc -l)

if [ $TRAIN_PROCS -gt 0 ]; then
    echo "⚠️  Found $TRAIN_PROCS training processes running"
    echo "   These processes have been stuck for 34+ hours without progress"
    echo "   (MIOpen find database issue - no batches processed)"
    echo ""
    
    read -p "Stop current training and restart with checkpointing? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping current training processes..."
        pkill -f "python train_patched.py"
        sleep 3
        
        # Verify stopped
        REMAINING=$(ps aux | grep "python train" | grep -v grep | wc -l)
        if [ $REMAINING -gt 0 ]; then
            echo "⚠️  Some processes still running, forcing kill..."
            pkill -9 -f "python train_patched.py"
            sleep 2
        fi
        
        echo "✅ Training stopped"
        echo ""
    else
        echo "❌ Cancelled - training continues running"
        exit 0
    fi
else
    echo "✅ No training processes running"
    echo ""
fi

# Backup old log
if [ -f logs/training_production.log ]; then
    echo "📦 Backing up old log..."
    cp logs/training_production.log "logs/training_production_backup_$(date +%Y%m%d_%H%M%S).log"
fi

# Start standalone training with screen
echo "🚀 Starting standalone training with checkpointing..."
echo ""

./start_training_standalone.sh

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     Next Steps                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Monitor training:"
echo "   screen -r yolo_training"
echo ""
echo "2. Detach from screen (leave running):"
echo "   Ctrl+A then D"
echo ""
echo "3. Check status anytime:"
echo "   ./check_training_status.sh"
echo ""
echo "4. View checkpoints:"
echo "   ls -lh runs/detect/train_standalone/weights/"
echo ""

