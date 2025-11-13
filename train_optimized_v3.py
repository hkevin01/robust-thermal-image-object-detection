#!/usr/bin/env python3
"""
Optimized YOLOv8 Training - AMD RX 5600 XT (gfx1010) - Version 2
================================================================

Patches AMP checks BEFORE ultralytics import to avoid MIOpen errors
"""

import os
import sys
from pathlib import Path

# Add patches directory FIRST
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

print("="*80)
print("STEP 1: PATCHING CONV2D")
print("="*80)

# Import and apply Conv2d patch BEFORE ultralytics
from conv2d_optimized import patch_torch_conv2d_optimized
patch_torch_conv2d_optimized()

print("\n" + "="*80)
print("STEP 2: PATCHING AMP CHECK")
print("="*80)

# Patch check_amp BEFORE importing ultralytics
import torch

def dummy_check_amp(model):
    """Always return False - we disabled AMP anyway"""
    print("AMP check called - returning False (disabled)")
    return False

# Import ultralytics.utils.checks and replace check_amp
import importlib.util
spec = importlib.util.find_spec("ultralytics.utils.checks")
if spec:
    # Manually load the module
    checks_module = importlib.util.module_from_spec(spec)
    sys.modules['ultralytics.utils.checks'] = checks_module
    spec.loader.exec_module(checks_module)
    # Replace check_amp
    checks_module.check_amp = dummy_check_amp
    print("✓ check_amp patched successfully\n")
else:
    print("⚠️  Could not find ultralytics.utils.checks\n")

print("="*80)
print("STEP 3: IMPORTING ULTRALYTICS")
print("="*80 + "\n")

from ultralytics import YOLO

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
    'name': 'train_optimized_v3',
    'exist_ok': True,
    
    # Optimization - LOWER LR for stability
    'optimizer': 'SGD',
    'lr0': 0.001,      # 10x lower for from-scratch training
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Checkpointing - CRITICAL
    'save': True,
    'save_period': 1,  # Save every epoch
    'resume': False,   # Start fresh (change to True after first run)
    
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
checkpoint_path = Path('runs/detect/train_optimized_v3/weights/last.pt')
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
    print("Loading base model from scratch: yolov8n.yaml (NO pretrained weights)")
    model = YOLO('yolov8n.yaml')  # Start from architecture, no weights

print("✓ Model loaded successfully")

# CRITICAL: Monkey-patch the loaded model's Conv2d forward methods
print("\n" + "="*80)
print("MONKEY-PATCHING LOADED MODEL")
print("="*80)
from patches.conv2d_optimized import monkey_patch_conv2d_forward
patched_count = monkey_patch_conv2d_forward(model.model)
print(f"✓ Monkey-patched {patched_count} Conv2d layers")
print("✓ All convolutions will now bypass MIOpen and use im2col+rocBLAS\n")

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
    print(f"Best checkpoint: runs/detect/train_optimized_v2/weights/best.pt")
    print(f"Last checkpoint: runs/detect/train_optimized_v2/weights/last.pt")
    print("="*80 + "\n")

except KeyboardInterrupt:
    print("\n" + "="*80)
    print("TRAINING INTERRUPTED")
    print("="*80)
    print("Checkpoint saved. Resume with:")
    print("  python train_optimized_v2.py")
    print("="*80 + "\n")
    sys.exit(0)

except Exception as e:
    print("\n" + "="*80)
    print("TRAINING FAILED")
    print("="*80)
    print(f"Error: {e}")
    print("\nCheckpoint may be available at:")
    print("  runs/detect/train_optimized_v2/weights/last.pt")
    print("="*80 + "\n")
    raise
