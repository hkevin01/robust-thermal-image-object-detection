#!/usr/bin/env python3
"""
Standalone Training Script with Robust Checkpointing
Runs independently of VS Code using screen/tmux
"""
import os
import sys
import signal
from pathlib import Path

# Add patches directory to path
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

# CRITICAL: Apply Conv2d patch BEFORE importing torch or ultralytics
print("="*80)
print("APPLYING CONV2D PATCH - MIOpen Bypass")
print("="*80)

from conv2d_fallback import patch_torch_conv2d
patch_torch_conv2d()

print("\n" + "="*80)
print("Patch applied successfully - importing ultralytics")
print("="*80 + "\n")

# Now import after patch
from ultralytics import YOLO
import torch

# Additional MIOpen environment variables
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\n⚠️ Received signal {signum}, saving checkpoint...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def train():
    print("="*80)
    print("🚀 Standalone YOLOv8 Training with Robust Checkpointing")
    print("="*80)
    
    # Check for existing checkpoint to resume from
    last_checkpoint = Path("runs/detect/train_standalone/weights/last.pt")
    
    if last_checkpoint.exists():
        print(f"📂 Found checkpoint: {last_checkpoint}")
        print("▶️ Resuming training from checkpoint...")
        model = YOLO(str(last_checkpoint))
    else:
        print("🆕 Starting fresh training from YOLOv8n pretrained weights")
        model = YOLO('yolov8n.pt')
    
    # Training configuration
    results = model.train(
        data='data/ltdv2_full/data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,
        workers=8,
        device=0,
        project='runs/detect',
        name='train_standalone',
        exist_ok=True,
        resume=last_checkpoint.exists(),
        
        # Optimization
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
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
        
        # Checkpointing
        save=True,
        save_period=1,  # Save checkpoint every epoch
        
        # Performance
        amp=False,  # Disable AMP for stability
        verbose=True,
        plots=True,
    )
    
    print("="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    return results

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
