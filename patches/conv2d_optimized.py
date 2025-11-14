"""
Optimized Conv2d Fallback - Tuned for AMD RX 5600 XT (gfx1010/RDNA1)
=====================================================================

Hardware: AMD Radeon RX 5600 XT (Navi 10, gfx1010 with HSA override to 10.3.0)
- 18 Compute Units
- 5.98 GB VRAM
- ROCm 5.2.21151
- PyTorch 1.13.1+rocm5.2

Issue: MIOpen lacks kernel database for gfx1010/gfx1030
Solution: Pure PyTorch im2col + rocBLAS matmul (bypasses MIOpen completely)

Optimizations Applied:
1. Cached output dimension calculations
2. Minimized tensor reshaping operations
3. In-place operations where possible
4. Optimized memory layout for rocBLAS
5. Batch-aware processing
6. Reduced memory footprint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import functools


# Cache for output dimension calculations
@functools.lru_cache(maxsize=128)
def _calc_output_dims(in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, 
                      padding_h, padding_w, dilation_h, dilation_w):
    """Cached output dimension calculation"""
    out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    return out_h, out_w


def im2col_conv2d_optimized(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Optimized im2col + matmul convolution for RDNA1 GPUs
    
    Optimizations:
    - Contiguous memory layouts for rocBLAS
    - Minimal reshaping operations
    - Cached dimension calculations
    - Efficient grouped convolution handling
    """
    
    # Normalize parameters
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Use cached output dimension calculation
    out_height, out_width = _calc_output_dims(
        in_height, in_width, kernel_h, kernel_w,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1]
    )
    
    if groups == 1:
        # Optimized path for standard convolution
        
        # Unfold - ensure contiguous for rocBLAS
        input_unfolded = F.unfold(
            input,
            kernel_size=(kernel_h, kernel_w),
            dilation=dilation,
            padding=padding,
            stride=stride
        ).contiguous()  # [batch, in_ch*kh*kw, num_patches]
        
        # Reshape weight - ensure contiguous
        weight_reshaped = weight.view(out_channels, -1).contiguous()
        
        # Matrix multiplication (rocBLAS backend)
        # Transpose for better memory access pattern
        output = torch.matmul(
            weight_reshaped,  # [out_ch, in_ch*kh*kw]
            input_unfolded    # [batch, in_ch*kh*kw, num_patches]
        )  # -> [batch, out_ch, num_patches]
        
        # Reshape to spatial dimensions
        output = output.view(batch_size, out_channels, out_height, out_width)
        
    else:
        # Grouped convolution - process groups separately
        channels_per_group_in = in_channels // groups
        channels_per_group_out = out_channels // groups
        
        # Pre-allocate output tensor
        output = torch.empty(
            batch_size, out_channels, out_height, out_width,
            dtype=input.dtype, device=input.device
        )
        
        for g in range(groups):
            # Slice indices
            in_start = g * channels_per_group_in
            in_end = (g + 1) * channels_per_group_in
            out_start = g * channels_per_group_out
            out_end = (g + 1) * channels_per_group_out
            
            # Extract group slices (avoid copy if possible)
            input_g = input[:, in_start:in_end, :, :]
            weight_g = weight[out_start:out_end, :, :, :]
            
            # Apply convolution to group
            input_unfolded_g = F.unfold(
                input_g,
                kernel_size=(kernel_h, kernel_w),
                dilation=dilation,
                padding=padding,
                stride=stride
            ).contiguous()
            
            weight_reshaped_g = weight_g.view(channels_per_group_out, -1).contiguous()
            
            output_g = torch.matmul(weight_reshaped_g, input_unfolded_g)
            
            # Write directly to output slice (in-place)
            output[:, out_start:out_end, :, :] = output_g.view(
                batch_size, channels_per_group_out, out_height, out_width
            )
    
    # Add bias if provided (in-place operation)
    if bias is not None:
        output.add_(bias.view(1, -1, 1, 1))
    
    return output


class OptimizedFallbackConv2d(nn.Conv2d):
    """
    Optimized drop-in replacement for nn.Conv2d
    
    Features:
    - Contiguous memory layouts
    - Minimal tensor copies
    - Efficient rocBLAS usage
    - Progress tracking for debugging
    """
    
    _instance_count = 0
    _forward_count = 0
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        OptimizedFallbackConv2d._instance_count += 1
        self.instance_id = OptimizedFallbackConv2d._instance_count
        
        # Only print for first few instances to reduce noise
        if self.instance_id <= 10 or self.instance_id % 50 == 0:
            print(f"[OptConv2d #{self.instance_id}] {self.in_channels}→{self.out_channels}, "
                  f"k={self.kernel_size}, s={self.stride}, p={self.padding}, g={self.groups}")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass"""
        
        # Track forward passes (for debugging stuck training)
        OptimizedFallbackConv2d._forward_count += 1
        
        # Handle non-zero padding modes
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        
        # Use optimized implementation
        return im2col_conv2d_optimized(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups
        )
    
    @classmethod
    def get_stats(cls):
        """Get usage statistics"""
        return {
            'total_instances': cls._instance_count,
            'total_forward_calls': cls._forward_count,
            'avg_calls_per_instance': cls._forward_count / max(cls._instance_count, 1)
        }


def patch_torch_conv2d_optimized():
    """
    Apply optimized Conv2d patch globally
    
    This replaces nn.Conv2d with our optimized fallback implementation.
    Call BEFORE importing any models.
    """
    print("\n" + "="*70)
    print("APPLYING OPTIMIZED CONV2D PATCH")
    print("="*70)
    print("Hardware: AMD RX 5600 XT (gfx1010/Navi 10)")
    print("Strategy: im2col (unfold) + rocBLAS (matmul)")
    print("Bypasses: MIOpen completely")
    print("="*70 + "\n")
    
    nn.Conv2d = OptimizedFallbackConv2d
    
    print("✓ Patch applied successfully")
    print("✓ All Conv2d layers will use optimized fallback\n")


def patch_model_conv2d_layers(model):
    """
    Patch all Conv2d layers in an already-loaded model
    
    This replaces existing nn.Conv2d instances with OptimizedFallbackConv2d.
    Use this when you need to patch a model loaded from a checkpoint.
    Handles both direct Conv2d and wrapped Conv2d (e.g., ultralytics Conv.conv).
    
    Args:
        model: PyTorch model or nn.Module
        
    Returns:
        int: Number of layers patched
    """
    import copy
    
    patched_count = 0
    
    def _recursive_patch(module, name=''):
        nonlocal patched_count
        
        # Check if this module has a 'conv' attribute (ultralytics Conv wrapper)
        if hasattr(module, 'conv') and isinstance(module.conv, nn.Conv2d) and not isinstance(module.conv, OptimizedFallbackConv2d):
            child_module = module.conv
            
            # Create optimized version with same parameters
            optimized = OptimizedFallbackConv2d(
                in_channels=child_module.in_channels,
                out_channels=child_module.out_channels,
                kernel_size=child_module.kernel_size,
                stride=child_module.stride,
                padding=child_module.padding,
                dilation=child_module.dilation,
                groups=child_module.groups,
                bias=child_module.bias is not None,
                padding_mode=child_module.padding_mode
            )
            
            # Copy weights and biases
            optimized.weight.data = child_module.weight.data.clone()
            if child_module.bias is not None:
                optimized.bias.data = child_module.bias.data.clone()
            
            # Replace the .conv attribute
            module.conv = optimized
            patched_count += 1
        
        # Iterate through immediate children
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # If this is a direct Conv2d, replace it
            if isinstance(child_module, nn.Conv2d) and not isinstance(child_module, OptimizedFallbackConv2d):
                # Create optimized version with same parameters
                optimized = OptimizedFallbackConv2d(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    padding_mode=child_module.padding_mode
                )
                
                # Copy weights and biases
                optimized.weight.data = child_module.weight.data.clone()
                if child_module.bias is not None:
                    optimized.bias.data = child_module.bias.data.clone()
                
                # Replace the module
                setattr(module, child_name, optimized)
                patched_count += 1
            else:
                # Recursively patch children
                _recursive_patch(child_module, full_name)
    
    _recursive_patch(model)
    return patched_count


# Progress monitoring for long operations
class ConvProgressMonitor:
    """Monitor Conv2d operations to detect hangs"""
    
    def __init__(self, check_interval=100):
        self.check_interval = check_interval
        self.last_count = 0
        self.stall_count = 0
    
    def check(self):
        """Check if forward passes are progressing"""
        current = OptimizedFallbackConv2d._forward_count
        
        if current == self.last_count:
            self.stall_count += 1
            if self.stall_count > 5:
                print(f"⚠️  WARNING: No Conv2d forward passes in last {self.stall_count * self.check_interval} checks")
        else:
            self.stall_count = 0
        
        self.last_count = current
        return current


# Testing and validation
def test_optimized_conv2d():
    """Comprehensive test suite for optimized Conv2d"""
    
    print("\n" + "="*70)
    print("TESTING OPTIMIZED CONV2D IMPLEMENTATION")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    test_cases = [
        # (name, in_ch, out_ch, kernel, stride, padding, groups)
        ("3x3 conv", 3, 64, 3, 1, 1, 1),
        ("3x3 stride 2", 64, 128, 3, 2, 1, 1),
        ("1x1 conv", 128, 256, 1, 1, 0, 1),
        ("3x3 grouped", 128, 128, 3, 1, 1, 4),
        ("Large batch", 256, 256, 3, 1, 1, 1),
    ]
    
    all_passed = True
    
    for name, in_ch, out_ch, kernel, stride, padding, groups in test_cases:
        print(f"Test: {name}")
        print(f"  Config: {in_ch}→{out_ch}, k={kernel}, s={stride}, p={padding}, g={groups}")
        
        try:
            # Create input
            batch = 8 if "Large" in name else 4
            x = torch.randn(batch, in_ch, 64, 64, device=device, requires_grad=True)
            
            # Create layer
            conv = OptimizedFallbackConv2d(
                in_ch, out_ch, kernel_size=kernel,
                stride=stride, padding=padding, groups=groups
            ).to(device)
            
            # Forward pass
            start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
            end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
            
            if device == 'cuda':
                start.record()
            
            output = conv(x)
            
            if device == 'cuda':
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                print(f"  ✓ Forward: {x.shape} → {output.shape} ({elapsed:.2f}ms)")
            else:
                print(f"  ✓ Forward: {x.shape} → {output.shape}")
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            # Verify gradients
            assert x.grad is not None, "Input gradients missing"
            assert conv.weight.grad is not None, "Weight gradients missing"
            print(f"  ✓ Backward: Gradients computed")
            
            # Check for NaN
            if torch.isnan(output).any():
                print(f"  ✗ FAILED: NaN in output")
                all_passed = False
            elif torch.isnan(conv.weight.grad).any():
                print(f"  ✗ FAILED: NaN in weight gradients")
                all_passed = False
            else:
                print(f"  ✓ No NaN values\n")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    test_optimized_conv2d()


class PicklableConvForward:
    """Picklable wrapper for optimized conv forward"""
    def __init__(self, stride, padding, dilation, groups):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def __call__(self, input, weight, bias):
        return im2col_conv2d_optimized(
            input=input,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


def monkey_patch_conv2d_forward(model):
    """
    Monkey-patch the forward method of all Conv2d layers in a model
    
    This replaces the _conv_forward method at runtime to use our optimized implementation.
    Uses a picklable wrapper class for checkpoint compatibility.
    
    Args:
        model: PyTorch model or nn.Module
        
    Returns:
        int: Number of layers patched
    """
    patched_count = 0
    
    def _recursive_patch(module):
        nonlocal patched_count
        
        # Patch direct Conv2d instances
        if isinstance(module, nn.Conv2d) and not isinstance(module, OptimizedFallbackConv2d):
            # Replace with picklable wrapper
            module._conv_forward = PicklableConvForward(
                module.stride, module.padding, module.dilation, module.groups
            )
            patched_count += 1
        
        # Check for wrapped Conv2d (e.g., ultralytics Conv.conv)
        if hasattr(module, 'conv') and isinstance(module.conv, nn.Conv2d):
            if not isinstance(module.conv, OptimizedFallbackConv2d):
                module.conv._conv_forward = PicklableConvForward(
                    module.conv.stride, module.conv.padding, module.conv.dilation, module.conv.groups
                )
                patched_count += 1
        
        # Recursively patch children
        for child in module.children():
            _recursive_patch(child)
    
    _recursive_patch(model)
    return patched_count
