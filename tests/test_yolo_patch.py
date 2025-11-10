#!/usr/bin/env python3
"""
Quick test: Verify YOLOv8 works with Conv2d patch
"""

import sys
from pathlib import Path

# Add patches to path
sys.path.insert(0, str(Path(__file__).parent / 'patches'))

# Apply patch
print("Applying Conv2d patch...")
from conv2d_fallback import patch_torch_conv2d
patch_torch_conv2d()

# Import ultralytics
print("Loading YOLOv8...")
from ultralytics import YOLO
import torch

# Create model
model = YOLO('yolov8n.pt')
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
