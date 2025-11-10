#!/bin/bash
# Training Dashboard - Complete overview

clear
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        YOLOv8 TRAINING DASHBOARD - MIOpen Bypass Solution          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Training Process Status
echo "📊 PROCESS STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ps aux | grep -q "[p]ython.*train_patched"; then
    TRAIN_PID=$(ps aux | grep "[p]ython.*train_patched" | head -1 | awk '{print $2}')
    RUNTIME=$(ps -p $TRAIN_PID -o etime= | tr -d ' ')
    CPU=$(ps -p $TRAIN_PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $TRAIN_PID -o %mem= | tr -d ' ')
    echo "  Status: ✅ RUNNING"
    echo "  PID: $TRAIN_PID"
    echo "  Runtime: $RUNTIME"
    echo "  CPU: ${CPU}%"
    echo "  Memory: ${MEM}%"
else
    echo "  Status: ❌ NOT RUNNING"
fi
echo ""

# GPU Status
echo "🎮 GPU STATUS (AMD RX 5600 XT - RDNA1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EDGE_TEMP=$(rocm-smi --showtemp 2>/dev/null | grep "edge" | awk '{print $NF}')
JUNCTION_TEMP=$(rocm-smi --showtemp 2>/dev/null | grep "junction" | awk '{print $NF}')
GPU_USE=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | awk '{print $NF}')
VRAM_USED=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Used" | awk '{print $NF}')
VRAM_TOTAL=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | head -1 | awk '{print $NF}')

VRAM_USED_GB=$(echo "scale=2; $VRAM_USED / 1073741824" | bc 2>/dev/null || echo "N/A")
VRAM_TOTAL_GB=$(echo "scale=2; $VRAM_TOTAL / 1073741824" | bc 2>/dev/null || echo "N/A")

echo "  Edge Temp: ${EDGE_TEMP}°C"
echo "  Junction Temp: ${JUNCTION_TEMP}°C"
echo "  Utilization: ${GPU_USE}%"
echo "  VRAM: ${VRAM_USED_GB}GB / ${VRAM_TOTAL_GB}GB"
echo ""

# Training Progress
echo "🎯 TRAINING PROGRESS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LATEST_PROGRESS=$(grep -E "^[[:space:]]*[0-9]+/[0-9]+" training_production.log 2>/dev/null | tail -1)
if [ -n "$LATEST_PROGRESS" ]; then
    echo "  $LATEST_PROGRESS"
else
    echo "  No progress data yet (initializing...)"
fi
echo ""

# Loss Trends (last 5 updates)
echo "📈 LOSS TRENDS (Last 5 Updates)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -E "^[[:space:]]*[0-9]+/[0-9]+" training_production.log 2>/dev/null | tail -5 | awk '{
    printf "  Epoch %s | Box: %s | Cls: %s | DFL: %s\n", $1, $3, $4, $5
}' || echo "  No loss data yet"
echo ""

# MIOpen Status
echo "🔧 MIOpen BYPASS STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MIOPEN_ERRORS=$(grep -i "miopenstatus\|miopen.*error" training_production.log 2>/dev/null | wc -l)
if [ "$MIOPEN_ERRORS" -eq 0 ]; then
    echo "  ✅ No MIOpen errors detected"
    echo "  ✅ Pure PyTorch fallback working correctly"
else
    echo "  ⚠️  $MIOPEN_ERRORS MIOpen errors found in log"
fi
echo ""

# Files Generated
echo "📁 OUTPUT FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "runs/detect/train2" ]; then
    echo "  Training dir: runs/detect/train2/"
    if [ -f "runs/detect/train2/weights/last.pt" ]; then
        LAST_SIZE=$(du -h runs/detect/train2/weights/last.pt 2>/dev/null | awk '{print $1}')
        echo "  Latest checkpoint: last.pt ($LAST_SIZE)"
    fi
    if [ -f "runs/detect/train2/results.csv" ]; then
        RESULTS_LINES=$(wc -l < runs/detect/train2/results.csv 2>/dev/null)
        echo "  Results CSV: $RESULTS_LINES entries"
    fi
else
    echo "  No output directory yet"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Monitoring Commands:                                              ║"
echo "║  • ./check_status.sh          - Quick status                       ║"
echo "║  • ./extract_metrics.sh       - Extract CSV metrics                ║"
echo "║  • tail -f training_production.log  - Live log                     ║"
echo "║  • tail -f training_monitor.log     - Monitor log                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
