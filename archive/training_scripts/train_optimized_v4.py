#!/usr/bin/env python3
"""
Optimized YOLOv8 Training - AMD RX 5600 XT (gfx1010) - Version 4
================================================================

KEY FIX: Properly configure model for 4 classes (not 80)
"""

import os
import sys
from pathlib import Path

# Add patches directory FIRST
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

print("="*80)
print("AMD RX 5600 XT - Training Version 4: FIXED CLASS MISMATCH")
print("="*80)

# Import Conv2d patch
from conv2d_optimized import monkey_patch_conv2d_forward
from ultralytics import YOLO

# Set environment variables
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_DISABLE_CACHE'] = '0'

def main():
    print("\nLoading model...")
    
    # Load model - ultralytics should auto-adjust for correct nc
    model = YOLO('yolov8n.pt')  # Pretrained on COCO (80 classes)
    
    print("\nApplying Conv2d monkey-patch...")
    # Apply monkey-patch BEFORE moving to CUDA
    num_patched = monkey_patch_conv2d_forward(model.model)
    print(f"✓ Patched {num_patched} Conv2d layers")
    
    print("\nTraining configuration:")
    print("  Model: yolov8n.pt")
    print("  Data: LTDv2 (4 classes)")  
    print("  Batch: 16")
    print("  Epochs: 50")
    print("  Learning rate: 0.001")
    print("  AMP: Disabled")
    print("  Workers: 4")
    
    # Train - ultralytics will rebuild model head for 4 classes
    results = model.train(
        data='data/ltdv2_full/data.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        device='cuda:0',
        workers=4,
        amp=False,
        project='runs/detect',
        name='train_optimized_v4',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
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
        copy_paste=0.0,
        save=True,
        save_period=-1,
        cache=False,
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
