#!/bin/bash

# MIOpen workaround for RDNA1 (gfx1030)
# Forces GEMM-based convolution which is more stable

export MIOPEN_FIND_MODE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=0
export MIOPEN_FIND_ENFORCE=3

# Disable problematic convolution algorithms
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_FFT=0

# Force GEMM convolution (most compatible)
export MIOPEN_DEBUG_CONV_GEMM=1

# Enable detailed logging
export MIOPEN_LOG_LEVEL=3

# Disable caching to force recompute
export MIOPEN_DISABLE_CACHE=0

echo "=== MIOpen Configuration ==="
echo "FIND_MODE: $MIOPEN_FIND_MODE"
echo "GEMM: $MIOPEN_DEBUG_CONV_GEMM"
echo "LOG_LEVEL: $MIOPEN_LOG_LEVEL"
echo ""

# Activate venv
source venv-py310-rocm52/bin/activate

# Start training with conservative settings
yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  batch=4 \
  device=0 \
  imgsz=640 \
  amp=False \
  workers=4 \
  name=production_yolov8n_rocm52_gemm \
  project=runs/detect
