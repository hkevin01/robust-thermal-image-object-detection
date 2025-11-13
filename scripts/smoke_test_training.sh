#!/bin/bash
# Smoke test: train on small subset to measure speed
# Creates a temporary data.yaml with reduced dataset

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              Smoke Test: Speed Baseline Check                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Create temporary small dataset config
mkdir -p data/smoke_test
cat > data/smoke_test/data.yaml << 'YAML'
# Smoke test dataset - subset of LTDv2
path: ../ltdv2_full
train: images/train
val: images/val

# Classes
names:
  0: person
  1: bicycle
  2: motorcycle
  3: vehicle
nc: 4
YAML

echo "✅ Created smoke test config"
echo ""

# Create smoke test training script
cat > scripts/train_smoke_test.py << 'PYEOF'
#!/usr/bin/env python3
"""Smoke test training - measure speed on small subset"""
import os
import sys
import random
from pathlib import Path

# Add patches
sys.path.insert(0, str(Path(__file__).parent.parent / 'patches'))

print("="*80)
print("APPLYING CONV2D PATCH - MIOpen Bypass")
print("="*80)

from conv2d_fallback import patch_torch_conv2d
patch_torch_conv2d()

print("\n" + "="*80)
print("Patch applied - importing ultralytics")
print("="*80 + "\n")

from ultralytics import YOLO
import torch

# MIOpen bypass
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'

print("🔍 Smoke Test Configuration:")
print("  - Epochs: 1")
print("  - Batch: 4")
print("  - Images: Limited to speed test")
print("  - Purpose: Measure realistic iteration speed")
print("")

model = YOLO('yolov8n.pt')

results = model.train(
    data='data/smoke_test/data.yaml',
    epochs=1,
    batch=4,
    imgsz=640,
    workers=8,
    device=0,
    project='runs/detect',
    name='smoke_test',
    exist_ok=True,
    
    # Optimization
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    # Checkpointing
    save=True,
    save_period=1,
    
    # Performance
    amp=False,
    verbose=True,
    plots=False,  # Skip plots for speed
    
    # Limit batches for quick test
    patience=0,
)

print("\n" + "="*80)
print("✅ Smoke test completed!")
print("="*80)
PYEOF

chmod +x scripts/train_smoke_test.py

echo "📝 Smoke test script created: scripts/train_smoke_test.py"
echo ""
echo "To run smoke test (will stop main training):"
echo "  1. screen -X -S yolo_training quit"
echo "  2. python scripts/train_smoke_test.py"
echo ""
echo "Or run in new screen:"
echo "  screen -dmS smoke_test bash -c 'cd ~/Projects/robust-thermal-image-object-detection && source venv-py310-rocm52/bin/activate && python scripts/train_smoke_test.py 2>&1 | tee logs/smoke_test_$(date +%Y%m%d_%H%M%S).log'"
echo ""
