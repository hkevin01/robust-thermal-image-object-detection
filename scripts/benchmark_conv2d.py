#!/usr/bin/env python3
"""
Conv2d Fallback Performance Benchmark
======================================

Measures performance of optimized Conv2d fallback on AMD RX 5600 XT
Compares against baseline and identifies bottlenecks
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add patches directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'patches'))

from conv2d_optimized import (
    OptimizedFallbackConv2d,
    patch_torch_conv2d_optimized,
    test_optimized_conv2d
)


def benchmark_single_conv(
    in_channels, out_channels, kernel_size,
    stride, padding, groups,
    input_size=(4, 3, 224, 224),
    num_iterations=10,
    warmup=3
):
    """Benchmark a single convolution configuration"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create input
    batch, in_ch, h, w = input_size
    x = torch.randn(batch, in_channels, h, w, device=device)
    
    # Create convolution layer
    conv = OptimizedFallbackConv2d(
        in_channels, out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups
    ).to(device)
    
    # Warmup
    for _ in range(warmup):
        _ = conv(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    times_forward = []
    for _ in range(num_iterations):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = conv(x)
            end.record()
            torch.cuda.synchronize()
            times_forward.append(start.elapsed_time(end))
        else:
            start = time.time()
            output = conv(x)
            end = time.time()
            times_forward.append((end - start) * 1000)
    
    # Benchmark backward pass
    times_backward = []
    x.requires_grad_(True)
    
    for _ in range(num_iterations):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            output = conv(x)
            loss = output.sum()
            
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            times_backward.append(start.elapsed_time(end))
            
            # Clear gradients
            conv.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
        else:
            output = conv(x)
            loss = output.sum()
            
            start = time.time()
            loss.backward()
            end = time.time()
            times_backward.append((end - start) * 1000)
            
            conv.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
    
    return {
        'forward_mean': sum(times_forward) / len(times_forward),
        'forward_std': torch.tensor(times_forward).std().item(),
        'backward_mean': sum(times_backward) / len(times_backward),
        'backward_std': torch.tensor(times_backward).std().item(),
        'total_mean': (sum(times_forward) + sum(times_backward)) / len(times_forward),
        'output_shape': tuple(output.shape)
    }


def run_yolo_representative_benchmark():
    """Benchmark Conv2d patterns typical in YOLOv8n"""
    
    print("\n" + "="*70)
    print("YOLOV8N REPRESENTATIVE CONV2D BENCHMARK")
    print("="*70 + "\n")
    
    # Apply patch
    patch_torch_conv2d_optimized()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Memory: {mem_gb:.2f} GB\n")
    
    # YOLOv8n typical layers (batch=4, 640x640 input)
    configs = [
        # (name, in_ch, out_ch, kernel, stride, padding, groups, input_size)
        ("Backbone Conv1", 3, 16, 3, 2, 1, 1, (4, 3, 640, 640)),
        ("Backbone Conv2", 16, 32, 3, 2, 1, 1, (4, 16, 320, 320)),
        ("CSP Block", 32, 32, 3, 1, 1, 1, (4, 32, 160, 160)),
        ("Bottleneck", 64, 64, 3, 1, 1, 1, (4, 64, 80, 80)),
        ("Neck Conv", 128, 128, 1, 1, 0, 1, (4, 128, 40, 40)),
        ("Head Conv", 256, 256, 3, 1, 1, 1, (4, 256, 20, 20)),
    ]
    
    print(f"{'Layer':<20} {'Input Shape':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15}")
    print("-" * 90)
    
    total_time = 0
    
    for name, in_ch, out_ch, k, s, p, g, input_size in configs:
        results = benchmark_single_conv(
            in_ch, out_ch, k, s, p, g,
            input_size=input_size,
            num_iterations=5,
            warmup=2
        )
        
        fwd = results['forward_mean']
        bwd = results['backward_mean']
        tot = results['total_mean']
        total_time += tot
        
        input_str = f"{input_size[0]}x{input_size[1]}x{input_size[2]}x{input_size[3]}"
        print(f"{name:<20} {input_str:<20} {fwd:>12.2f}    {bwd:>12.2f}    {tot:>12.2f}")
    
    print("-" * 90)
    print(f"{'TOTAL':<20} {'':<20} {'':<15} {'':<15} {total_time:>12.2f}\n")
    
    # Estimate batches per second
    batches_per_sec = 1000.0 / total_time  # total_time is in ms
    print(f"Estimated throughput: {batches_per_sec:.3f} batches/second")
    print(f"Time per batch: {total_time:.2f} ms")
    print(f"YOLOv8n forward+backward: ~{total_time:.0f}ms per batch\n")
    
    # Estimate training time
    total_batches = 82325  # LTDv2 train set
    epochs = 50
    total_iterations = total_batches * epochs
    
    time_per_batch_sec = total_time / 1000.0
    total_time_sec = total_iterations * time_per_batch_sec
    total_time_hours = total_time_sec / 3600
    total_time_days = total_time_hours / 24
    
    print("="*70)
    print("TRAINING TIME ESTIMATE")
    print("="*70)
    print(f"Batches per epoch: {total_batches}")
    print(f"Total epochs: {epochs}")
    print(f"Total iterations: {total_iterations}")
    print(f"Time per batch: {time_per_batch_sec:.3f} seconds")
    print(f"Estimated total time: {total_time_hours:.1f} hours ({total_time_days:.1f} days)")
    print("="*70 + "\n")
    
    return total_time


def run_stress_test():
    """Stress test to verify stability over extended period"""
    
    print("\n" + "="*70)
    print("STABILITY STRESS TEST")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a small test network
    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = OptimizedFallbackConv2d(3, 16, 3, padding=1)
            self.conv2 = OptimizedFallbackConv2d(16, 32, 3, padding=1)
            self.conv3 = OptimizedFallbackConv2d(32, 64, 3, padding=1)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            return x
    
    model = SmallNet().to(device)
    x = torch.randn(4, 3, 128, 128, device=device, requires_grad=True)
    
    num_iterations = 100
    print(f"Running {num_iterations} iterations...")
    
    nan_detected = False
    errors = []
    
    for i in range(num_iterations):
        try:
            output = model(x)
            loss = output.sum()
            loss.backward()
            
            # Check for NaN
            if torch.isnan(output).any():
                print(f"  ✗ NaN detected at iteration {i+1}")
                nan_detected = True
                break
            
            # Clear gradients
            model.zero_grad()
            x.grad.zero_()
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ Iteration {i+1}/{num_iterations} - OK")
        
        except Exception as e:
            print(f"  ✗ Error at iteration {i+1}: {e}")
            errors.append((i+1, str(e)))
            break
    
    print()
    if nan_detected:
        print("✗ STRESS TEST FAILED - NaN detected")
        return False
    elif errors:
        print(f"✗ STRESS TEST FAILED - {len(errors)} errors")
        return False
    else:
        print("✓ STRESS TEST PASSED - All iterations completed successfully")
        return True


def main():
    """Run all benchmarks"""
    
    print("\n" + "="*70)
    print("CONV2D FALLBACK COMPREHENSIVE BENCHMARK SUITE")
    print("Hardware: AMD RX 5600 XT (gfx1010)")
    print("="*70)
    
    # Test basic functionality
    print("\n[1/3] Running basic functionality tests...")
    if not test_optimized_conv2d():
        print("✗ Basic tests failed. Aborting benchmark.")
        return 1
    
    # Benchmark performance
    print("\n[2/3] Running performance benchmark...")
    avg_time = run_yolo_representative_benchmark()
    
    # Stress test
    print("\n[3/3] Running stability stress test...")
    stable = run_stress_test()
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Basic tests: ✓ PASSED")
    print(f"Performance: {avg_time:.2f}ms per batch (YOLOv8n representative)")
    print(f"Stability: {'✓ PASSED' if stable else '✗ FAILED'}")
    print("="*70 + "\n")
    
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
