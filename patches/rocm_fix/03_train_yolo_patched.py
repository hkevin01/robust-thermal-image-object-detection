#!/usr/bin/env python3
"""
YOLO Training with ROCm RDNA1/2 Fix Applied
Wrapper script that applies all memory coherency patches before training.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print("=" * 80)
print("YOLO Training with ROCm RDNA1/2 Memory Coherency Fix")
print("=" * 80)
print(f"Project root: {project_root}\n")

# Import and apply the fix FIRST
from patches.rocm_fix.hip_memory_patch import apply_rocm_fix
fix = apply_rocm_fix()

# Now import ultralytics and other modules
print("\n📦 Importing ultralytics...")
from ultralytics import YOLO
import torch
import argparse

print(f"✓ Ultralytics imported successfully")
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def train_yolo_safe(
    data_yaml='data/ltdv2_full/data.yaml',
    model='yolov8n.pt',
    epochs=10,
    batch_size=4,
    imgsz=640,
    device='0',
    workers=4,
    name='baseline_yolov8n_patched',
    **kwargs
):
    """
    Train YOLO with ROCm fix applied.
    
    Args:
        data_yaml: Path to dataset YAML file
        model: Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Batch size (smaller is safer for RDNA1/2)
        imgsz: Input image size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        workers: Number of data loading workers
        name: Training run name
        **kwargs: Additional YOLO arguments
    """
    print("\n" + "=" * 80)
    print(f"Starting YOLO Training (PATCHED)")
    print("=" * 80)
    print(f"  Model:      {model}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch:      {batch_size}")
    print(f"  Image Size: {imgsz}")
    print(f"  Device:     {device}")
    print(f"  Workers:    {workers}")
    print(f"  Name:       {name}")
    print("=" * 80)
    
    try:
        # Load model
        print(f"\n1. Loading model: {model}")
        yolo_model = YOLO(model)
        print(f"   ✓ Model loaded successfully")
        
        # Train with safe parameters
        print(f"\n2. Starting training...")
        results = yolo_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            workers=workers,
            name=name,
            patience=20,
            save=True,
            save_period=10,
            cache='ram',  # Cache in RAM, not GPU
            amp=False,  # Disable AMP (can cause issues)
            verbose=True,
            **kwargs
        )
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return results
        
    except RuntimeError as e:
        error_str = str(e).lower()
        if "memory access fault" in error_str or "hip error" in error_str:
            print("\n" + "=" * 80)
            print("❌ MEMORY ACCESS FAULT DETECTED")
            print("=" * 80)
            print(f"Error: {e}\n")
            print("The patch was unable to prevent the memory fault.")
            print("\nRecommended actions:")
            print("1. Apply kernel module fix: sudo ./patches/rocm_fix/01_kernel_params.sh")
            print("2. Reboot system")
            print("3. Try again with smaller batch size")
            print("4. Use CPU training as fallback: device='cpu'")
            print("=" * 80)
            raise
        else:
            print(f"\n❌ Training error: {e}")
            raise
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description='Train YOLO with ROCm RDNA1/2 fix')
    parser.add_argument('--data', default='data/ltdv2_full/data.yaml', help='Dataset YAML')
    parser.add_argument('--model', default='yolov8n.pt', help='Model size')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--name', default='baseline_yolov8n_patched', help='Run name')
    
    args = parser.parse_args()
    
    # Run training
    train_yolo_safe(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name
    )


if __name__ == "__main__":
    main()
