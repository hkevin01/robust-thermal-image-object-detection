#!/usr/bin/env python3
"""
Optimized YOLOv8 Training - AMD RX 5600 XT (gfx1010) - Version 4 FIXED
==================================================================

KEY FIX: Re-apply monkey-patch after model rebuild for correct class count
"""

import os
import sys
from pathlib import Path

# Add patches directory FIRST
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

print("="*80)
print("AMD RX 5600 XT - Training Version 4 FIXED: CLASS COUNT + RE-PATCH")
print("="*80)

# Import Conv2d patch
from conv2d_optimized import monkey_patch_conv2d_forward
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# Set environment variables
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_DISABLE_CACHE'] = '0'

# Monkey-patch the trainer to re-apply Conv2d patch after model rebuild
original_setup_model = BaseTrainer.setup_model

def patched_setup_model(self):
    """Wrap setup_model to re-apply Conv2d monkey-patch after model rebuild"""
    result = original_setup_model(self)
    
    # Re-apply monkey-patch AFTER model is set up with correct nc
    print("\n" + "="*80)
    print("MODEL REBUILT - RE-APPLYING CONV2D MONKEY-PATCH")
    print("="*80)
    num_patched = monkey_patch_conv2d_forward(self.model)
    print(f"✓ Re-patched {num_patched} Conv2d layers")
    print(f"✓ Model now configured for {self.model.model[-1].nc} classes")
    print("="*80 + "\n")
    
    return result

BaseTrainer.setup_model = patched_setup_model

def main():
    print("\nLoading pretrained model...")
    model = YOLO('yolov8n.pt')  # Will be rebuilt for 4 classes during training
    
    print("\nTraining configuration:")
    print("  Model: yolov8n.pt (will rebuild head for 4 classes)")
    print("  Data: LTDv2 (4 classes)")  
    print("  Batch: 8  (reduced from 16 to avoid OOM)")
    print("  Epochs: 50")
    print("  Learning rate: 0.001")
    print("  AMP: Disabled")
    print("  Workers: 0  (single-threaded to avoid DataLoader deadlock)")
    
    # Train - our patched trainer will re-apply monkey-patch after model rebuild
    results = model.train(
        data='data/ltdv2_full/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cuda:0',
        workers=0,  # Single-threaded to avoid DataLoader deadlock with ROCm
        amp=False,
        project='runs/detect',
        name='train_optimized_v4_fixed',
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
        save=True,
        save_period=-1,
        cache=False,
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        val=True,
        plots=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
