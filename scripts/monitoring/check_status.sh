#!/bin/bash
echo "================================"
echo "🚀 TRAINING STATUS CHECK"
echo "================================"
echo ""
echo "📊 Process Status:"
ps aux | grep "python.*train_patched" | grep -v grep | head -3 | awk '{printf "  PID: %s | CPU: %s%% | MEM: %s%% | Time: %s\n", $2, $3, $4, $10}'
echo ""
echo "�� GPU Status:"
rocm-smi --showuse --showmeminfo vram --showtemp | grep -E "Temperature|GPU use|VRAM Total Used"
echo ""
echo "📝 Recent Log (last 10 lines):"
tail -10 training_production.log | sed 's/^/  /'
echo ""
echo "================================"
