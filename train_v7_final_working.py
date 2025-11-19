#!/usr/bin/env python3
"""
YOLOv8 Training - Final Working Version
Workers=0, AMP disabled, patches applied
"""

import os
import sys
from datetime import datetime
import torch
import torch.nn

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

# Patch training step to add gradient clipping for NaN prevention
from ultralytics.engine.trainer import BaseTrainer
original_optimizer_step = BaseTrainer.optimizer_step

def patched_optimizer_step(self):
    """Add gradient clipping before optimizer step"""
    # Clip gradients to prevent explosion
    if hasattr(self, 'scaler'):
        self.scaler.unscale_(self.optimizer)
    
    # Gradient clipping - STRENGTHENED for Epoch 4+ stability
    max_norm = 5.0  # More aggressive clipping (reduced from 10.0)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
    
    # Check for NaN/Inf in gradients before stepping
    params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
    if params_with_grad:
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in params_with_grad]))
        if not torch.isfinite(grad_norm):
            print(f"⚠️  WARNING: Non-finite gradient detected (norm={grad_norm}), skipping optimizer step")
            self.optimizer.zero_grad()
            return
    
    # Proceed with original optimizer step
    original_optimizer_step(self)

BaseTrainer.optimizer_step = patched_optimizer_step
print("✓ Gradient clipping patched into training loop")

print("="*70)
print("🚀 Starting Training - NaN PREVENTION STRENGTHENED v2 + VERBOSE")
print("="*70)
print("  • Workers: 0 (FORCED IN MULTIPLE PLACES)")
print("  • AMP: Disabled (ROCm compatibility)")
print("  • Resume: From last.pt checkpoint")
print("  • Gradient Clipping: STRENGTHENED (max_norm=5.0)")
print("  • NaN Detection: ENABLED (auto-skip bad steps)")
print("  • Learning Rate: ULTRA-REDUCED (0.00025 from 0.0005)")
print("  • Warmup: EXTENDED (10 epochs from 5)")
print("  • Momentum: REDUCED (0.85 from 0.9)")
print("  • Weight Decay: INCREASED (0.001 from 0.0005)")
print("  • Validation: ENABLED (every epoch)")
print("  • VERBOSE LOGGING: ENABLED")
print("="*70 + "\n")

print(f"[{datetime.now().strftime('%H:%M:%S')}] About to call model.train()...")
sys.stdout.flush()

# Add simple callback for progress monitoring
def on_train_epoch_start(trainer):
    """Print when epoch starts"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️  Starting Epoch {trainer.epoch + 1}/{trainer.epochs}", flush=True)
    except:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️  Starting new epoch", flush=True)

def on_train_epoch_end(trainer):
    """Print when epoch ends"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Completed Epoch {trainer.epoch + 1}/{trainer.epochs}", flush=True)
    except:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Epoch completed", flush=True)

# Register callbacks
model.add_callback("on_train_epoch_start", on_train_epoch_start)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Progress callbacks registered")
sys.stdout.flush()

# Train with NaN prevention measures
print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling model.train() with parameters...")
sys.stdout.flush()

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
    resume=False,  # START FRESH - don't load old hyperparameters from checkpoint
    val=True,  # Enable validation to catch issues earlier
    amp=False,
    save=True,
    save_period=1,
    verbose=True,  # Force verbose output
    
    # ULTRA-REDUCED learning rate to prevent gradient explosion (STRENGTHENED)
    lr0=0.00025,  # Further reduced from 0.0005 (50% reduction)
    lrf=0.01,
    
    # More conservative optimizer settings (STRENGTHENED)
    momentum=0.85,  # Further reduced from 0.9
    weight_decay=0.001,  # Increased from 0.0005 (stronger regularization)
    optimizer='SGD',  # Explicit SGD (more stable than Adam with NaNs)
    
    # EXTENDED warmup for stability (STRENGTHENED)
    warmup_epochs=10.0,  # Doubled from 5.0 epochs
    warmup_momentum=0.8,
    warmup_bias_lr=0.025,  # Further reduced from 0.05
    
    # Conservative data augmentation (STRENGTHENED - reduce potential for extreme values)
    hsv_h=0.005,  # Further reduced from 0.01
    hsv_s=0.3,    # Further reduced from 0.5
    hsv_v=0.2,    # Further reduced from 0.3
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ Training completed!")
sys.stdout.flush()
