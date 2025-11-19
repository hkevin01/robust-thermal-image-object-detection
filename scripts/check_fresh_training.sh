#!/bin/bash
echo "=== Training Status ==="
ps aux | grep train_v7_final_working | grep -v grep
echo ""
echo "=== Latest Progress ==="
tail -5 logs/training_FRESH_START_STRENGTHENED_*.log
echo ""
echo "=== NaN Count ==="
grep -c "Non-finite gradient" logs/training_FRESH_START_STRENGTHENED_*.log || echo "0"

