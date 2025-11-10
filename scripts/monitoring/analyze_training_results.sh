#!/bin/bash

# Detailed Training Results Analysis Script
# Analyzes results.csv and provides comprehensive metrics

RESULTS_FILE="runs/detect/production_yolov8n_rocm522/results.csv"
WEIGHTS_DIR="runs/detect/production_yolov8n_rocm522/weights"

echo "=========================================="
echo "YOLOv8 Training Results Analysis"
echo "=========================================="
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "❌ Results file not found: $RESULTS_FILE"
    echo "   Epoch 1 has not completed yet."
    echo "   This analysis will be available after the first epoch finishes."
    exit 1
fi

# Get total epochs completed
total_epochs=$(tail -n +2 "$RESULTS_FILE" | wc -l)
echo "📊 Total Epochs Completed: $total_epochs / 100"
echo ""

# Training runtime
if ps -p 407777 > /dev/null 2>&1; then
    runtime=$(ps -p 407777 -o etime= | tr -d ' ')
    echo "⏱️  Training Runtime: $runtime"
else
    echo "⏱️  Training Runtime: Completed or stopped"
fi
echo ""

# Parse CSV header to get column indices
header=$(head -1 "$RESULTS_FILE")
IFS=',' read -ra COLUMNS <<< "$header"

# Find column indices (adjust based on actual CSV structure)
epoch_col=0
box_loss_col=2
cls_loss_col=3
dfl_loss_col=4
precision_col=5
recall_col=6
map50_col=9
map5095_col=10

echo "=========================================="
echo "📈 TRAINING METRICS SUMMARY"
echo "=========================================="
echo ""

# Latest epoch metrics
echo "🔹 Latest Epoch (Epoch $total_epochs):"
latest=$(tail -1 "$RESULTS_FILE")
IFS=',' read -ra VALUES <<< "$latest"

printf "   Box Loss:      %.4f\n" "${VALUES[$box_loss_col]}"
printf "   Class Loss:    %.4f\n" "${VALUES[$cls_loss_col]}"
printf "   DFL Loss:      %.4f\n" "${VALUES[$dfl_loss_col]}"
if [ "${VALUES[$precision_col]}" != "" ]; then
    printf "   Precision:     %.4f\n" "${VALUES[$precision_col]}"
    printf "   Recall:        %.4f\n" "${VALUES[$recall_col]}"
    printf "   mAP@0.5:       %.4f\n" "${VALUES[$map50_col]}"
    printf "   mAP@0.5:0.95:  %.4f\n" "${VALUES[$map5095_col]}"
fi
echo ""

# First epoch comparison
echo "🔹 First Epoch (Epoch 1):"
first=$(head -2 "$RESULTS_FILE" | tail -1)
IFS=',' read -ra FIRST_VALUES <<< "$first"

printf "   Box Loss:      %.4f\n" "${FIRST_VALUES[$box_loss_col]}"
printf "   Class Loss:    %.4f\n" "${FIRST_VALUES[$cls_loss_col]}"
printf "   DFL Loss:      %.4f\n" "${FIRST_VALUES[$dfl_loss_col]}"
echo ""

# Calculate improvements
if [ "$total_epochs" -gt 1 ]; then
    echo "�� Improvement (First → Latest):"
    
    box_loss_first=${FIRST_VALUES[$box_loss_col]}
    box_loss_latest=${VALUES[$box_loss_col]}
    box_improvement=$(awk "BEGIN {printf \"%.2f\", ($box_loss_first - $box_loss_latest) / $box_loss_first * 100}")
    printf "   Box Loss:      %.1f%% reduction\n" "$box_improvement"
    
    cls_loss_first=${FIRST_VALUES[$cls_loss_col]}
    cls_loss_latest=${VALUES[$cls_loss_col]}
    cls_improvement=$(awk "BEGIN {printf \"%.2f\", ($cls_loss_first - $cls_loss_latest) / $cls_loss_first * 100}")
    printf "   Class Loss:    %.1f%% reduction\n" "$cls_improvement"
    
    dfl_loss_first=${FIRST_VALUES[$dfl_loss_col]}
    dfl_loss_latest=${VALUES[$dfl_loss_col]}
    dfl_improvement=$(awk "BEGIN {printf \"%.2f\", ($dfl_loss_first - $dfl_loss_latest) / $dfl_loss_first * 100}")
    printf "   DFL Loss:      %.1f%% reduction\n" "$dfl_improvement"
    echo ""
fi

# Best epoch analysis
echo "=========================================="
echo "🏆 BEST PERFORMANCE"
echo "=========================================="
echo ""

# Find best mAP50 epoch
if [ "${VALUES[$map50_col]}" != "" ]; then
    best_map50_line=$(tail -n +2 "$RESULTS_FILE" | awk -F',' -v col=$((map50_col+1)) '{print $col, NR}' | sort -rn | head -1)
    best_map50_value=$(echo "$best_map50_line" | awk '{print $1}')
    best_map50_epoch=$(echo "$best_map50_line" | awk '{print $2}')
    
    echo "🔹 Best mAP@0.5:"
    printf "   Epoch %d: %.4f\n" "$best_map50_epoch" "$best_map50_value"
    echo ""
    
    # Get full metrics for best epoch
    best_epoch_data=$(tail -n +2 "$RESULTS_FILE" | sed -n "${best_map50_epoch}p")
    IFS=',' read -ra BEST_VALUES <<< "$best_epoch_data"
    
    echo "   Full metrics at best epoch:"
    printf "   ├─ Box Loss:      %.4f\n" "${BEST_VALUES[$box_loss_col]}"
    printf "   ├─ Class Loss:    %.4f\n" "${BEST_VALUES[$cls_loss_col]}"
    printf "   ├─ DFL Loss:      %.4f\n" "${BEST_VALUES[$dfl_loss_col]}"
    printf "   ├─ Precision:     %.4f\n" "${BEST_VALUES[$precision_col]}"
    printf "   ├─ Recall:        %.4f\n" "${BEST_VALUES[$recall_col]}"
    printf "   └─ mAP@0.5:0.95:  %.4f\n" "${BEST_VALUES[$map5095_col]}"
fi
echo ""

# Loss trends
echo "=========================================="
echo "📉 LOSS TRENDS (Last 10 Epochs)"
echo "=========================================="
echo ""
echo "Epoch | Box Loss | Cls Loss | DFL Loss"
echo "------|----------|----------|----------"
tail -10 "$RESULTS_FILE" | awk -F',' -v b=$((box_loss_col+1)) -v c=$((cls_loss_col+1)) -v d=$((dfl_loss_col+1)) '{
    printf "%5d | %8.4f | %8.4f | %8.4f\n", $1, $b, $c, $d
}'
echo ""

# mAP trends
if [ "${VALUES[$map50_col]}" != "" ]; then
    echo "=========================================="
    echo "📈 mAP TRENDS (Last 10 Epochs)"
    echo "=========================================="
    echo ""
    echo "Epoch | mAP@0.5  | mAP@0.5:0.95 | Precision | Recall"
    echo "------|----------|--------------|-----------|--------"
    tail -10 "$RESULTS_FILE" | awk -F',' -v m50=$((map50_col+1)) -v m5095=$((map5095_col+1)) -v p=$((precision_col+1)) -v r=$((recall_col+1)) '{
        printf "%5d | %8.4f | %12.4f | %9.4f | %6.4f\n", $1, $m50, $m5095, $p, $r
    }'
    echo ""
fi

# Model weights status
echo "=========================================="
echo "💾 MODEL WEIGHTS"
echo "=========================================="
echo ""

if [ -d "$WEIGHTS_DIR" ]; then
    if [ -f "$WEIGHTS_DIR/best.pt" ]; then
        best_size=$(ls -lh "$WEIGHTS_DIR/best.pt" | awk '{print $5}')
        best_time=$(stat -c %y "$WEIGHTS_DIR/best.pt" | cut -d. -f1)
        echo "✅ best.pt:  $best_size (updated: $best_time)"
    else
        echo "⏳ best.pt:  Not yet created"
    fi
    
    if [ -f "$WEIGHTS_DIR/last.pt" ]; then
        last_size=$(ls -lh "$WEIGHTS_DIR/last.pt" | awk '{print $5}')
        last_time=$(stat -c %y "$WEIGHTS_DIR/last.pt" | cut -d. -f1)
        echo "✅ last.pt:  $last_size (updated: $last_time)"
    else
        echo "⏳ last.pt:  Not yet created"
    fi
else
    echo "⚠️  Weights directory not found"
fi
echo ""

# Performance assessment
echo "=========================================="
echo "🎯 PERFORMANCE ASSESSMENT"
echo "=========================================="
echo ""

if [ "${VALUES[$map50_col]}" != "" ]; then
    current_map50="${VALUES[$map50_col]}"
    
    # Use awk for floating point comparison
    if awk "BEGIN {exit !($current_map50 >= 0.7)}"; then
        echo "🟢 Excellent: mAP@0.5 >= 0.7 (Target achieved!)"
    elif awk "BEGIN {exit !($current_map50 >= 0.5)}"; then
        echo "🟡 Good: mAP@0.5 >= 0.5 (On track)"
    elif awk "BEGIN {exit !($current_map50 >= 0.3)}"; then
        echo "🟠 Fair: mAP@0.5 >= 0.3 (Still learning)"
    elif awk "BEGIN {exit !($current_map50 >= 0.1)}"; then
        echo "🔵 Early: mAP@0.5 >= 0.1 (Early training stage)"
    else
        echo "⚪ Starting: mAP@0.5 < 0.1 (Model initialization)"
    fi
    echo ""
    
    # Predictions
    if [ "$total_epochs" -lt 100 ]; then
        echo "📊 Progress: $total_epochs% complete"
        remaining=$((100 - total_epochs))
        echo "⏳ Remaining: $remaining epochs"
        echo ""
    fi
fi

# System health
echo "=========================================="
echo "🔧 SYSTEM HEALTH CHECKS"
echo "=========================================="
echo ""

# Check for errors
error_count=$(grep -c -i "error\|exception" training.log 2>/dev/null || echo "0")
hsa_error_count=$(grep -c "HSA_STATUS_ERROR" training.log 2>/dev/null || echo "0")

if [ "$error_count" -eq 0 ]; then
    echo "✅ No errors in training log"
else
    echo "⚠️  $error_count errors found in log"
fi

if [ "$hsa_error_count" -eq 0 ]; then
    echo "✅ No HSA errors (IMPLICIT_GEMM working)"
else
    echo "❌ $hsa_error_count HSA errors (RDNA1 bug!)"
fi

# Check process
if ps -p 407777 > /dev/null 2>&1; then
    echo "✅ Training process running (PID 407777)"
    cpu=$(ps -p 407777 -o %cpu= | tr -d ' ')
    mem=$(ps -p 407777 -o %mem= | tr -d ' ')
    echo "   CPU: ${cpu}% | Memory: ${mem}%"
else
    echo "⚠️  Training process not running"
fi
echo ""

# Output files
echo "=========================================="
echo "📁 OUTPUT FILES"
echo "=========================================="
echo ""
ls -lh runs/detect/production_yolov8n_rocm522/*.{jpg,png,csv} 2>/dev/null | awk '{print $9, "("$5")"}'
echo ""

# Recommendations
echo "=========================================="
echo "💡 RECOMMENDATIONS"
echo "=========================================="
echo ""

if [ "$total_epochs" -lt 10 ]; then
    echo "• Continue training - too early to evaluate"
    echo "• Check back after 10 epochs for initial assessment"
elif [ "$total_epochs" -lt 50 ]; then
    if [ "${VALUES[$map50_col]}" != "" ]; then
        if awk "BEGIN {exit !($current_map50 < 0.3)}"; then
            echo "⚠️  mAP@0.5 lower than expected at this stage"
            echo "• Review training images (train_batch*.jpg)"
            echo "• Check if dataset labels are correct"
        else
            echo "✅ Training progressing normally"
            echo "• Continue to 50 epochs for midpoint assessment"
        fi
    fi
elif [ "$total_epochs" -ge 50 ] && [ "$total_epochs" -lt 100 ]; then
    if [ "${VALUES[$map50_col]}" != "" ]; then
        if awk "BEGIN {exit !($current_map50 >= 0.7)}"; then
            echo "🎉 Excellent performance! Consider:"
            echo "• Training may have converged early"
            echo "• You could stop now or continue for fine-tuning"
        elif awk "BEGIN {exit !($current_map50 >= 0.5)}"; then
            echo "✅ Good progress! Continue to 100 epochs"
        else
            echo "⚠️  mAP@0.5 below target at midpoint"
            echo "• Review hyperparameters if final mAP@0.5 < 0.5"
            echo "• Consider longer training or larger model"
        fi
    fi
elif [ "$total_epochs" -ge 100 ]; then
    echo "🎉 Training complete! Next steps:"
    echo "• Run validation: yolo detect val model=weights/best.pt"
    echo "• Test on unseen images: yolo detect predict model=weights/best.pt source=test/"
    echo "• Document results in ROCM_52_TRAINING_SUCCESS.md"
    echo "• Create backup: tar -czf training_backup.tar.gz runs/detect/production_yolov8n_rocm522"
fi
echo ""

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
