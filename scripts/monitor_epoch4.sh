#!/bin/bash
# Monitor Epoch 4 Progress and NaN Detection

echo "=== Epoch 4 Monitoring Dashboard ==="
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
tail -20 logs/training_NAN_PREVENTION_20251117.log | grep -E "4/50.*batch" | tail -1
echo ""

# NaN count (should be 0 for clean Epoch 4)
echo "⚠️  NaN Detection:"
NAN_COUNT=$(grep -c " nan " logs/training_NAN_PREVENTION_20251117.log 2>/dev/null || echo "0")
echo "Total NaN occurrences in log: $NAN_COUNT"
echo ""

# Gradient clipping warnings
GRADIENT_WARNINGS=$(grep -c "Non-finite gradient detected" logs/training_NAN_PREVENTION_20251117.log 2>/dev/null || echo "0")
echo "🛡️  Gradient Clipping Interventions: $GRADIENT_WARNINGS"
echo ""

# Memory usage
echo "💾 GPU Memory:"
tail -20 logs/training_NAN_PREVENTION_20251117.log | grep -E "GPU_mem" | tail -1
echo ""

echo "---"
echo "Refresh: watch -n 30 ./monitor_epoch4.sh"
