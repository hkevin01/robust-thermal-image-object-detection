#!/bin/bash
# Extract training metrics from log file

LOG_FILE="training_production.log"
OUTPUT="training_metrics.csv"

echo "Extracting training metrics from $LOG_FILE..."

# Extract epoch lines with metrics
echo "epoch,gpu_mem,box_loss,cls_loss,dfl_loss,instances,size,progress" > "$OUTPUT"

grep -E "^[[:space:]]*[0-9]+/[0-9]+" "$LOG_FILE" | while IFS= read -r line; do
    # Parse the line (example: "      1/50      2.05G      2.325      4.714      1.601         64        640: 0% ──── 302/82325")
    echo "$line" | awk '{
        epoch=$1
        gpu_mem=$2
        box_loss=$3
        cls_loss=$4
        dfl_loss=$5
        instances=$6
        size=$7
        gsub(/:/, "", $8)
        progress=$8
        printf "%s,%s,%s,%s,%s,%s,%s,%s\n", epoch, gpu_mem, box_loss, cls_loss, dfl_loss, instances, size, progress
    }'
done >> "$OUTPUT"

LINE_COUNT=$(wc -l < "$OUTPUT")
echo "Extracted $((LINE_COUNT - 1)) metric entries to $OUTPUT"

# Show last 5 entries
echo ""
echo "Latest metrics:"
tail -5 "$OUTPUT" | column -t -s,
