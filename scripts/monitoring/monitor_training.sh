#!/bin/bash
# Continuous Training Monitor
# Logs GPU health, training progress, and metrics every 5 minutes

LOG_FILE="training_monitor.log"
INTERVAL=300  # 5 minutes

echo "Starting training monitor (logging every 5 minutes)..."
echo "Monitor log: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "======================================" >> "$LOG_FILE"
    echo "Timestamp: $TIMESTAMP" >> "$LOG_FILE"
    echo "======================================" >> "$LOG_FILE"
    
    # Check if training is still running
    if ! ps aux | grep -q "[p]ython.*train_patched"; then
        echo "❌ Training process not found!" >> "$LOG_FILE"
        echo "Training may have completed or crashed."
        echo "Check training_production.log for details."
        break
    fi
    
    # GPU Temperature
    echo -e "\n📊 GPU Status:" >> "$LOG_FILE"
    rocm-smi --showtemp --showuse --showmeminfo vram 2>/dev/null | grep -E "Temperature|GPU use|VRAM Total Used" >> "$LOG_FILE"
    
    # Training Progress (last line with epoch info)
    echo -e "\n🎯 Training Progress:" >> "$LOG_FILE"
    grep -E "^[[:space:]]*[0-9]+/[0-9]+" training_production.log | tail -1 >> "$LOG_FILE"
    
    # Check for errors
    if grep -i "error\|exception\|failed" training_production.log | tail -5 | grep -qi "error"; then
        echo -e "\n⚠️  Recent errors detected:" >> "$LOG_FILE"
        grep -i "error\|exception" training_production.log | tail -3 >> "$LOG_FILE"
    fi
    
    # Display to console
    echo "[$TIMESTAMP] ✓ Logged - GPU: $(rocm-smi --showtemp 2>/dev/null | grep 'edge' | awk '{print $NF}')°C"
    
    sleep $INTERVAL
done
