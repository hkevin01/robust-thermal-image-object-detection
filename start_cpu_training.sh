#!/bin/bash
# CPU Training Script for RDNA1 workaround

source venv-py310-rocm52/bin/activate

echo "=========================================="
echo "  YOLOv8 CPU Training (RDNA1 Workaround)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Epochs: 30"
echo "  Batch: 16"
echo "  Image Size: 416"
echo "  Workers: 8"
echo "  Device: CPU"
echo ""
echo "Expected Timeline:"
echo "  Per Epoch: ~10-12 hours"
echo "  Total: ~10-15 days"
echo ""
echo "Monitor with: tail -f training_cpu.log"
echo "Check status: ps -p \$(cat .training_cpu_pid)"
echo ""
echo "Starting training..."

nohup yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=30 \
  batch=16 \
  device=cpu \
  imgsz=416 \
  workers=8 \
  name=production_yolov8n_cpu \
  project=runs/detect \
  > training_cpu.log 2>&1 &

echo $! > .training_cpu_pid
sleep 2

echo ""
echo "✅ Training started!"
echo "PID: $(cat .training_cpu_pid)"
echo ""
echo "First epoch will take ~12 hours to see initial progress."
echo "Be patient - CPU training is slow but will complete eventually."
