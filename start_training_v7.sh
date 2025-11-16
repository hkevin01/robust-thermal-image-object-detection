#!/bin/bash
LOG_FILE="training_v7_$(date +%Y%m%d_%H%M%S).log"
python -u train_optimized_v7_rocm_fixes.py 2>&1 | tee "$LOG_FILE"
