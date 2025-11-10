#!/bin/bash

echo "=========================================="
echo "YOLOv8 Training Progress Monitor"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if training process is running
if ps aux | grep -q "[4]07777"; then
    echo "✅ Training Status: RUNNING"
    runtime=$(ps -p 407777 -o etime= | tr -d ' ')
    echo "⏱️  Runtime: $runtime"
    echo ""
else
    echo "❌ Training Status: NOT RUNNING"
    echo "Check training.log for errors"
    exit 1
fi

# Check CPU and memory usage
echo "📊 Resource Usage:"
ps aux | grep "[4]07777" | head -1 | awk '{printf "   CPU: %s%% | RAM: %.1f GB\n", $3, $6/1024/1024}'
echo ""

# Check log file size and growth
logsize=$(ls -lh training.log | awk '{print $5}')
echo "📝 Log File: training.log ($logsize)"
echo ""

# Extract latest training progress
echo "🔄 Current Progress:"
tail -100 training.log | grep -E "[0-9]+/100.*[0-9]\.[0-9]+G" | tail -1 | sed 's/^/   /'
echo ""

# Check for completed epochs
completed_epochs=$(grep -c "Validating:" training.log 2>/dev/null || echo "0")
echo "✅ Completed Epochs: $completed_epochs / 100"
echo ""

# Check if results.csv exists
if [ -f "runs/detect/production_yolov8n_rocm522/results.csv" ]; then
    echo "📈 Latest Metrics (results.csv):"
    tail -1 runs/detect/production_yolov8n_rocm522/results.csv | awk -F',' '{
        printf "   Epoch: %s | box_loss: %.3f | cls_loss: %.3f\n", $1, $3, $4
        if ($10 != "") printf "   mAP50: %.3f | mAP50-95: %.3f\n", $10, $11
    }'
else
    echo "📈 Metrics: results.csv not yet created (after epoch 1)"
fi
echo ""

# Check for errors
errors=$(grep -c -i "error\|exception\|fail" training.log 2>/dev/null || echo "0")
if [ "$errors" -gt 0 ]; then
    echo "⚠️  Errors found: $errors (check training.log)"
else
    echo "✅ No errors detected"
fi
echo ""

# Check for HSA errors (RDNA1 specific)
hsa_errors=$(grep -c "HSA_STATUS_ERROR" training.log 2>/dev/null || echo "0")
if [ "$hsa_errors" -gt 0 ]; then
    echo "❌ HSA ERRORS FOUND: $hsa_errors (RDNA1 bug returned!)"
else
    echo "✅ No HSA errors (IMPLICIT_GEMM working)"
fi
echo ""

# Estimate completion time
if [ "$completed_epochs" -gt 0 ]; then
    runtime_seconds=$(ps -p 407777 -o etimes= | tr -d ' ')
    seconds_per_epoch=$((runtime_seconds / completed_epochs))
    remaining_epochs=$((100 - completed_epochs))
    remaining_seconds=$((seconds_per_epoch * remaining_epochs))
    remaining_hours=$((remaining_seconds / 3600))
    remaining_mins=$(((remaining_seconds % 3600) / 60))
    
    echo "⏰ Estimated Time Remaining: ${remaining_hours}h ${remaining_mins}m"
    completion_time=$(date -d "+${remaining_seconds} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
    echo "🏁 Estimated Completion: $completion_time"
fi

echo "=========================================="
