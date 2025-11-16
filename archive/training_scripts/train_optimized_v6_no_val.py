#!/usr/bin/env python3
"""
Optimized YOLOv8 Training - AMD RX 5600 XT (gfx1010) - Version 6 NO VALIDATION
================================================================================

KEY FEATURES:
1. Resume from checkpoint (last.pt from v5)
2. Disable validation to isolate hang issue
3. Keep spawn context + persistent workers
4. Enhanced logging
"""

import os
import sys
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

# Set multiprocessing start method to 'spawn' BEFORE importing torch
mp.set_start_method('spawn', force=True)

# Add patches directory FIRST
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

print("="*80)
print("AMD RX 5600 XT - Training Version 6: NO VALIDATION TEST")
print("="*80)
print(f"Multiprocessing context: {mp.get_start_method()}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Import Conv2d patch
from conv2d_optimized import monkey_patch_conv2d_forward
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import torch.utils.data

# Set environment variables
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_DISABLE_CACHE'] = '0'

# Monkey-patch DataLoader to use spawn context and persistent workers
original_dataloader_init = torch.utils.data.DataLoader.__init__

def patched_dataloader_init(self, *args, **kwargs):
    """Enhanced DataLoader with spawn context and persistent workers"""
    num_workers = kwargs.get('num_workers', 0)
    if num_workers > 0:
        if 'multiprocessing_context' not in kwargs:
            kwargs['multiprocessing_context'] = 'spawn'
            print(f"  ✓ DataLoader: workers={num_workers}, context='spawn'")
        
        if 'persistent_workers' not in kwargs:
            kwargs['persistent_workers'] = True
            print(f"  ✓ DataLoader: persistent_workers=True")
    
    return original_dataloader_init(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = patched_dataloader_init

# Monkey-patch the trainer to re-apply Conv2d patch after model rebuild
original_setup_model = BaseTrainer.setup_model

def patched_setup_model(self):
    """Wrap setup_model to re-apply Conv2d monkey-patch after model rebuild"""
    result = original_setup_model(self)
    
    print("\n" + "="*80)
    print(f"MODEL SETUP - RE-APPLYING CONV2D PATCHES [{datetime.now().strftime('%H:%M:%S')}]")
    print("="*80)
    num_patched = monkey_patch_conv2d_forward(self.model)
    print(f"✓ Re-patched {num_patched} Conv2d layers")
    print(f"✓ Model configured for {self.model.model[-1].nc} classes")
    print("="*80 + "\n")
    
    return result

BaseTrainer.setup_model = patched_setup_model

# Add epoch completion logging
original_train_epoch = BaseTrainer._do_train

def patched_train_epoch(self, world_size=1):
    """Add logging at epoch completion"""
    print(f"\n{'='*80}")
    print(f"STARTING EPOCH {self.epoch + 1}/{self.epochs} [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*80}\n")
    
    result = original_train_epoch(self, world_size)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED EPOCH {self.epoch}/{self.epochs} [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"✓ Checkpoint should be saved")
    print(f"{'='*80}\n")
    
    return result

# Don't patch _do_train as it's not easily patchable, use callbacks instead

def main():
    # Check for existing checkpoint
    checkpoint_path = 'runs/detect/train_optimized_v5_multiprocess/weights/last.pt'
    resume_from = None
    
    if Path(checkpoint_path).exists():
        print(f"\n✓ Found checkpoint: {checkpoint_path}")
        print(f"  Will resume from Epoch 1 (completed)")
        resume_from = checkpoint_path
    else:
        print("\nNo checkpoint found, starting fresh")
    
    print("\nLoading model...")
    if resume_from:
        model = YOLO(resume_from)
        print(f"✓ Loaded checkpoint from Epoch 1")
    else:
        model = YOLO('yolov8n.pt')
        print("✓ Loaded pretrained yolov8n.pt")
    
    print("\nTraining configuration:")
    print("  Model: Resume from last.pt (Epoch 1 complete)")
    print("  Data: LTDv2 (4 classes)")  
    print("  Batch: 8")
    print("  Epochs: 50 (will continue from Epoch 2)")
    print("  Learning rate: 0.001")
    print("  AMP: Disabled")
    print("  Workers: 4 (spawn context + persistent_workers)")
    print("  Validation: DISABLED (testing if this causes hang)")
    print("  Checkpoints: Save every epoch")
    
    # Train with validation DISABLED
    results = model.train(
        data='data/ltdv2_full/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cuda:0',
        workers=4,
        amp=False,
        project='runs/detect',
        name='train_optimized_v6_no_val',
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
        save_period=1,  # Save every epoch
        cache=False,
        exist_ok=True,  # Allow overwriting
        pretrained=True if not resume_from else False,
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        val=False,  # DISABLE VALIDATION
        plots=True,
        resume=True if resume_from else False,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
