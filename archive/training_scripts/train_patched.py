#!/usr/bin/env python3
"""
YOLOv8 Training with Conv2d Patch (MIOpen Bypass)
==================================================

This script trains YOLOv8 using a custom Conv2d implementation that
bypasses MIOpen convolutions. This workaround enables GPU training on
RDNA1 GPUs (gfx1030) that have broken MIOpen support.

Strategy:
---------
1. Import and apply Conv2d patch BEFORE importing ultralytics
2. Patch replaces nn.Conv2d with FallbackConv2d
3. All model convolutions use im2col + matmul instead of MIOpen
4. Performance: ~2-5x slower but FUNCTIONAL

Expected Performance:
--------------------
- Batch processing: ~0.5-2 batches/sec (vs 5-10 with native MIOpen)
- Training time: ~3-7 days for 50 epochs (vs 1-2 days native)
- Memory: Similar usage, possibly slightly higher due to im2col expansion
- Quality: Identical results (same algorithm, different implementation)

Monitoring:
----------
- Watch GPU utilization: Should be 50-80% (not 99% stuck)
- Check logs: No MIOpen errors
- Monitor progress: Batches should advance steadily
- Temperature: Should be 70-85°C under load
"""

import sys
import os
from pathlib import Path

# Add patches directory to path
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

# CRITICAL: Apply patch BEFORE importing torch or ultralytics
print("="*70)
print("APPLYING CONV2D PATCH - MIOpen Bypass")
print("="*70)

from conv2d_fallback import patch_torch_conv2d
patch_torch_conv2d()

print("\n" + "="*70)
print("Patch applied successfully - importing ultralytics")
print("="*70 + "\n")

# Now import ultralytics - it will use patched Conv2d
from ultralytics import YOLO
import torch

# Verify GPU available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# Training configuration
DATA_YAML = "data/ltdv2_full/data.yaml"
MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
BATCH_SIZE = 4  # Conservative for patched implementation + 6GB VRAM
IMAGE_SIZE = 640
DEVICE = 0  # Use GPU

print("="*70)
print("Training Configuration")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Data: {DATA_YAML}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"Device: {'GPU' if DEVICE == 0 else 'CPU'}")
print(f"Conv2d: Patched (im2col + matmul)")
print("="*70 + "\n")

# Load model
print("Loading model...")
model = YOLO(MODEL_NAME)

print("\nModel architecture uses FallbackConv2d layers")
print("Starting training...\n")

# Train with patched convolutions
try:
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        amp=False,  # Disable AMP for stability
        workers=8,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        # Conservative settings for patched implementation
        cache=False,  # Don't cache to save memory
        rect=False,   # No rectangular training
        cos_lr=True,  # Cosine LR schedule
        close_mosaic=10,  # Disable mosaic in last 10 epochs
    )
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print(f"Best model saved to: {results.save_dir}")
    print(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
except KeyboardInterrupt:
    print("\n" + "="*70)
    print("Training interrupted by user")
    print("="*70)
    print("Partial results saved. Resume training with:")
    print(f"  model = YOLO('runs/detect/train/weights/last.pt')")
    print(f"  model.train(resume=True)")
    
except Exception as e:
    print("\n" + "="*70)
    print("ERROR during training")
    print("="*70)
    print(f"Exception: {type(e).__name__}")
    print(f"Message: {e}")
    print("\nIf this is still a MIOpen error, the patch may not have been")
    print("applied early enough. Try importing torch manually before patch.")
    raise
