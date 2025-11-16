#!/usr/bin/env python3
"""
YOLOv8 Training - Single-Threaded Fallback (workers=0)
Version 7-ST: Guaranteed stable configuration

This is the fallback if multiprocessing fixes don't work.
Proven stable: ran 904 batches without issues.
"""

import os
import sys
from datetime import datetime

from ultralytics import YOLO

# Load model first
model = YOLO('yolov8n.pt')

# Import and apply Conv2d patch to the loaded model
from patches.conv2d_optimized import monkey_patch_conv2d_forward
num_patched = monkey_patch_conv2d_forward(model.model)
print(f"✓ Patched {num_patched} Conv2d layers with im2col + rocBLAS")

# Wrap setup_model to re-patch after model rebuild
from ultralytics.engine.trainer import BaseTrainer
original_setup_model = BaseTrainer.setup_model

def patched_setup_model(self):
    """Re-patch Conv2d layers after model is rebuilt"""
    result = original_setup_model(self)
    
    # Re-patch Conv2d layers
    num_repatched = monkey_patch_conv2d_forward(self.model)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Re-patched {num_repatched} Conv2d layers after model rebuild\n")
    
    return result

BaseTrainer.setup_model = patched_setup_model
print("✓ BaseTrainer.setup_model wrapped for re-patching")

print("\n" + "="*70)
print("🚀 Starting Training - Single-Threaded Mode")
print("="*70)
print(f"Configuration:")
print(f"  • Workers: 0 (single-threaded - GUARANTEED STABLE)")
print(f"  • Speed: 2.1 it/s (19% slower than workers=4)")
print(f"  • ETA: Nov 26 (4 days buffer)")
print(f"  • Validation: Disabled (val=False)")
print(f"  • Resume: From last.pt checkpoint")
print("="*70 + "\n")

# Train the model
results = model.train(
    data='data/ltdv2_full/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cuda:0',
    workers=0,  # Single-threaded for guaranteed stability
    project='runs/detect',
    name='train_optimized_v7_single_thread',
    exist_ok=True,
    resume=True,  # Resume from last checkpoint
    
    # Disable validation
    val=False,
    
    # Disable AMP (causes MIOpen errors with ROCm)
    amp=False,
    
    # Save checkpoints
    save=True,
    save_period=1,  # Save every epoch
    
    # Optimization parameters
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Augmentation
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

print("\n" + "="*70)
print("✅ Training completed successfully!")
print("="*70)
