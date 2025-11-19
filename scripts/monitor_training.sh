#!/bin/bash
# Monitor Training Progress

echo "=== YOLOv8n Training Dashboard ==="
echo ""

# Process status
echo "📊 Process Status:"
if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        ps -p $PID -o pid,etime,pcpu,pmem,cmd
    else
        echo "❌ Training process not running"
    fi
else
    echo "❌ No training.pid file"
fi
echo ""

# Current progress
echo "📈 Current Progress:"
LOGFILE=$(ls -t logs/training_STRENGTHENED_*.log 2>/dev/null | head -1 || echo "logs/training_NAN_PREVENTION_20251117.log")
LATEST=$(grep -E "[0-9]+/50.*it/s" "$LOGFILE" | tail -1)
if [ -n "$LATEST" ]; then
    echo "$LATEST"
else
    echo "Initializing..."
fi
echo ""

# NaN detection (should stay 0)
echo "🛡️  NaN Protection Status:"
NAN_COUNT=$(grep -c " nan " "$LOGFILE" 2>/dev/null || echo "0")
GRADIENT_WARNINGS=$(grep -c "Non-finite gradient detected" "$LOGFILE" 2>/dev/null || echo "0")
echo "NaN occurrences: $NAN_COUNT ✅"
echo "Gradient interventions: $GRADIENT_WARNINGS"
echo ""

# Recent losses
echo "📉 Recent Losses:"
grep -E "[0-9]+/50.*it/s" "$LOGFILE" | tail -3
echo ""
echo "📁 Log file: $LOGFILE"
echo ""

echo "---"
echo "Auto-refresh: watch -n 30 ./monitor_training.sh"
