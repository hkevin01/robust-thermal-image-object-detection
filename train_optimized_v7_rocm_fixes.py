#!/usr/bin/env python3
"""
YOLOv8 Training with ROCm DataLoader Fixes
Version 7: Simplified with proven multiprocessing fixes
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Set multiprocessing start method to 'spawn' BEFORE importing torch
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
    print("✓ Multiprocessing start method set to 'spawn'")
except RuntimeError as e:
    print(f"Warning: Could not set spawn method: {e}")

import torch
from ultralytics import YOLO
from datetime import datetime
from torch.utils.data import DataLoader

print(f"✓ Environment configured for ROCm")

# Monkey-patch DataLoader to apply ROCm fixes
original_dataloader_init = DataLoader.__init__

def patched_dataloader_init(self, *args, **kwargs):
    """Apply ROCm-specific DataLoader fixes"""
    num_workers = kwargs.get('num_workers', 0)
    
    if num_workers > 0:
        # Use spawn context
        kwargs['multiprocessing_context'] = 'spawn'
        
        # DISABLE persistent workers (causes resource leaks with ROCm)
        kwargs['persistent_workers'] = False
        
        # Reduce prefetch factor to minimize queued batches
        if 'prefetch_factor' not in kwargs:
            kwargs['prefetch_factor'] = 2
        
        print(f"  DataLoader: workers={num_workers}, persistent=False, prefetch={kwargs.get('prefetch_factor')}")
    
    return original_dataloader_init(self, *args, **kwargs)

DataLoader.__init__ = patched_dataloader_init
print("✓ DataLoader patched with ROCm fixes")

# Load model first
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# Import and apply Conv2d patch to the loaded model
from patches.conv2d_optimized import monkey_patch_conv2d_forward
num_patched = monkey_patch_conv2d_forward(model.model)
print(f"✓ Patched {num_patched} Conv2d layers")

# Wrap setup_model to re-patch after model rebuild
from ultralytics.engine.trainer import BaseTrainer
original_setup_model = BaseTrainer.setup_model

def patched_setup_model(self):
    result = original_setup_model(self)
    num_repatched = monkey_patch_conv2d_forward(self.model)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Re-patched {num_repatched} Conv2d layers\n")
    return result

BaseTrainer.setup_model = patched_setup_model

print("\n" + "="*70)
print("🚀 Training with ROCm Fixes")
print("="*70)
print("  • Workers: 2 | Persistent: False | Prefetch: 2")
print("  • Spawn context | Val: Disabled")
print("="*70 + "\n")

# Train
results = model.train(
    data='data/ltdv2_full/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cuda:0',
    workers=2,
    project='runs/detect',
    name='train_optimized_v7_rocm_fixes',
    exist_ok=True,
    resume=True,
    val=False,
    save=True,
    save_period=1,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    verbose=True,
)

print("\n✅ Training completed!")
