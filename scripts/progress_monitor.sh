#!/bin/bash
# Monitor training progress for N minutes (default 15)
DURATION_MIN=${1:-15}
OUT=logs/progress_monitor.log
LOG=$(ls -t logs/training_2025*.log 2>/dev/null | head -1)

echo "Progress monitor started: $(date)" > "$OUT"
echo "Monitoring log: $LOG" >> "$OUT"
echo "Duration: ${DURATION_MIN} minutes" >> "$OUT"
echo "---" >> "$OUT"

END=$(( $(date +%s) + DURATION_MIN*60 ))
while [ $(date +%s) -lt $END ]; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TS] Sample" >> "$OUT"
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        tail -n 5 "$LOG" >> "$OUT"
    else
        echo "No training log found" >> "$OUT"
    fi
    rocm-smi --showuse --showtemp | grep -E "GPU use|Temperature|edge" >> "$OUT" 2>/dev/null || true
    echo "---" >> "$OUT"
    sleep 60
done

# Summary
echo "Progress monitor completed: $(date)" >> "$OUT"

# Print short summary
echo "--- Monitor Summary ---"
Tail -n 20 "$OUT"
