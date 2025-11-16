#!/usr/bin/env python3
"""
YOLOv8 Training - Final Working Version
Workers=0, AMP disabled, patches applied
"""

import os
import sys
from datetime import datetime

# Add project root to path so checkpoint can find patches.conv2d_optimized module
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import conv2d_optimized early so checkpoint can unpickle it
import patches.conv2d_optimized
sys.modules['conv2d_optimized'] = patches.conv2d_optimized  # Map old name to new module

# Patch ultralytics check_amp to always return False for ROCm
from ultralytics.utils import checks
original_check_amp = checks.check_amp

def patched_check_amp(model):
    """Always return False for ROCm compatibility"""
    print("⚠️  AMP check bypassed for ROCm compatibility")
    return False

checks.check_amp = patched_check_amp
print("✓ AMP check patched")

from ultralytics import YOLO

# Load model from checkpoint to resume training
checkpoint_path = 'runs/detect/train_optimized_v5_multiprocess/weights/last.pt'
print(f"Loading checkpoint: {checkpoint_path}")
model = YOLO(checkpoint_path)

# Import and apply Conv2d patch to the loaded model
from patches.conv2d_optimized import monkey_patch_conv2d_forward
num_patched = monkey_patch_conv2d_forward(model.model)
print(f"✓ Patched {num_patched} Conv2d layers")

# Wrap setup_model to re-patch after model rebuild AND force workers=0
from ultralytics.engine.trainer import BaseTrainer
original_setup_model = BaseTrainer.setup_model

def patched_setup_model(self):
    result = original_setup_model(self)
    num_repatched = monkey_patch_conv2d_forward(self.model)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Re-patched {num_repatched} Conv2d layers\n")
    return result

BaseTrainer.setup_model = patched_setup_model

# CRITICAL: Force workers=0 in trainer initialization to override checkpoint config
original_init = BaseTrainer.__init__

def patched_init(self, cfg=None, overrides=None, _callbacks=None):
    # Force workers=0 before initialization
    if overrides is None:
        overrides = {}
    overrides['workers'] = 0
    print(f"🔧 FORCING workers=0 (overriding checkpoint config)")
    result = original_init(self, cfg=cfg, overrides=overrides, _callbacks=_callbacks)
    # Double-check it's set
    if hasattr(self, 'args') and hasattr(self.args, 'workers'):
        self.args.workers = 0
        print(f"✓ Confirmed: self.args.workers = {self.args.workers}")
    return result

BaseTrainer.__init__ = patched_init

# EXTRA SAFETY: Patch the build_dataloader to force num_workers=0
original_build_dataloader = None
try:
    from ultralytics.data import build_dataloader
    original_build_dataloader = build_dataloader
    
    def patched_build_dataloader(*args, **kwargs):
        kwargs['workers'] = 0
        if 'num_workers' in kwargs:
            kwargs['num_workers'] = 0
        return original_build_dataloader(*args, **kwargs)
    
    # Monkey patch at module level
    import ultralytics.data
    ultralytics.data.build_dataloader = patched_build_dataloader
    print("✓ Patched build_dataloader to force workers=0")
except Exception as e:
    print(f"⚠️  Could not patch build_dataloader: {e}")

print("="*70)
print("🚀 Starting Training - FINAL VERSION")
print("="*70)
print("  • Workers: 0 (FORCED IN MULTIPLE PLACES)")
print("  • AMP: Disabled (ROCm compatibility)")
print("  • Resume: From last.pt checkpoint")
print("="*70 + "\n")

# Train
results = model.train(
    data='data/ltdv2_full/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cuda:0',
    workers=0,
    project='runs/detect',
    name='train_v7_final_working',
    exist_ok=True,
    resume=True,
    val=False,
    amp=False,
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
