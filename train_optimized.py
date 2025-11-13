#!/usr/bin/env python3
"""
Optimized YOLOv8 Training - AMD RX 5600 XT (gfx1010)
====================================================

Uses optimized Conv2d fallback with im2col + rocBLAS
Expected performance: ~18 batches/sec, 62 hours for 50 epochs

Hardware: AMD RX 5600 XT (Navi 10, gfx1010)
- PyTorch 1.13.1+rocm5.2
- ROCm 5.2.21151
- 5.98 GB VRAM
"""

import os
import sys
from pathlib import Path

# Add patches directory FIRST
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

print("="*80)
print("APPLYING OPTIMIZED CONV2D PATCH")
print("="*80)

# Import and apply patch BEFORE ultralytics
from conv2d_optimized import patch_torch_conv2d_optimized
patch_torch_conv2d_optimized()

print("\n" + "="*80)
print("IMPORTING ULTRALYTICS")
print("="*80 + "\n")

from ultralytics import YOLO
import torch

# Monkey-patch AMP check to always return False (we disabled AMP anyway)
print("Disabling AMP checks (not compatible with Conv2d patch)...")
try:
    from ultralytics.utils import checks
    checks.check_amp = lambda model: False
    print("✓ AMP checks disabled\n")
except Exception as e:
    print(f"⚠️  Could not disable AMP checks: {e}\n")

# MIOpen bypass environment variables
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_DISABLE_CACHE'] = '0'

# GPU info
print("="*80)
print("SYSTEM CONFIGURATION")
print("="*80)
print(f"PyTorch: {torch.version.__version__ if hasattr(torch.version, '__version__') else 'N/A'}")
print(f"ROCm: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")
print("="*80 + "\n")

# Training configuration
CONFIG = {
    'data': 'data/ltdv2_full/data.yaml',
    'epochs': 50,
    'batch': 4,
    'imgsz': 640,
    'workers': 8,
    'device': 0,
    'project': 'runs/detect',
    'name': 'train_optimized',
    'exist_ok': True,
    
    # Optimization
    'optimizer': 'SGD',
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Checkpointing - CRITICAL
    'save': True,
    'save_period': 1,  # Save every epoch
    'resume': True,    # Auto-resume from last.pt
    
    # Performance
    'amp': False,      # Disable AMP for stability
    'verbose': True,
    'plots': True,
    
    # Augmentation
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
}

print("="*80)
print("TRAINING CONFIGURATION")
print("="*80)
for key, value in CONFIG.items():
    print(f"{key:20s}: {value}")
print("="*80 + "\n")

print("="*80)
print("ESTIMATED TRAINING TIME")
print("="*80)
print("Based on benchmark results:")
print("  • Time per batch: ~55ms")
print("  • Batches per epoch: 82,325")
print("  • Total epochs: 50")
print("  • Estimated total time: 62.6 hours (2.6 days)")
print("  • Expected completion: ~Nov 14-15, 2025")
print("="*80 + "\n")

# Check for checkpoint
checkpoint_path = Path('runs/detect/train_optimized/weights/last.pt')
if checkpoint_path.exists():
    print(f"✓ Found checkpoint: {checkpoint_path}")
    print("  Training will resume from last checkpoint\n")
else:
    print("✓ No checkpoint found - starting fresh training\n")

# Load model
print("="*80)
print("LOADING MODEL")
print("="*80)

if checkpoint_path.exists() and CONFIG['resume']:
    print(f"Loading from checkpoint: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
else:
    print("Loading base model: yolov8n.pt")
    model = YOLO('yolov8n.pt')

print("✓ Model loaded successfully\n")

# Start training
print("="*80)
print("STARTING TRAINING")
print("="*80)
print("Press Ctrl+C to stop gracefully (checkpoint will be saved)")
print("="*80 + "\n")

try:
    results = model.train(**CONFIG)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Best checkpoint: runs/detect/train_optimized/weights/best.pt")
    print(f"Last checkpoint: runs/detect/train_optimized/weights/last.pt")
    print("="*80 + "\n")

except KeyboardInterrupt:
    print("\n" + "="*80)
    print("TRAINING INTERRUPTED")
    print("="*80)
    print("Checkpoint saved. Resume with:")
    print("  python train_optimized.py")
    print("="*80 + "\n")
    sys.exit(0)

except Exception as e:
    print("\n" + "="*80)
    print("TRAINING FAILED")
    print("="*80)
    print(f"Error: {e}")
    print("\nCheckpoint may be available at:")
    print("  runs/detect/train_optimized/weights/last.pt")
    print("="*80 + "\n")
    raise
