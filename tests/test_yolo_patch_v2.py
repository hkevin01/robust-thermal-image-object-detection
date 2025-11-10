#!/usr/bin/env python3
"""
YOLOv8 Patch Test v2 - Patch after model loading
"""

import sys
from pathlib import Path

# Add patches to path
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

# Import ultralytics FIRST (let it create the model with standard Conv2d)
print("Loading YOLOv8 (with standard Conv2d)...")
from ultralytics import YOLO
import torch

# Create model
model = YOLO('yolov8n.pt')

# NOW apply patch to existing model
print("\nApplying patch to existing model...")
from conv2d_monkey_patch import apply_full_patch
apply_full_patch(model.model)

# Move to GPU
model.to('cuda')

print("\nTesting forward pass with patched convolutions...")

# Create dummy input
dummy_input = torch.randn(2, 3, 640, 640, device='cuda')

# Test forward pass
try:
    with torch.no_grad():
        output = model.model(dummy_input)
    print("✓ Forward pass successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output length: {len(output)}")
    print(f"  Output[0] shape: {output[0].shape}")
    
    # Test backward pass (training mode)
    model.train()
    dummy_input.requires_grad = True
    output = model.model(dummy_input)
    loss = output[0].sum()
    loss.backward()
    print("✓ Backward pass successful!")
    print(f"  Gradients computed: {dummy_input.grad is not None}")
    
    print("\n" + "="*70)
    print("SUCCESS: YOLOv8 works with Conv2d patch!")
    print("="*70)
    print("Ready to start full training.")
    
except Exception as e:
    print("\n" + "="*70)
    print("FAILED: Error during testing")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
