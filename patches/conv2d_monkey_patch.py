"""
Runtime Conv2d Monkey Patch - Replace forward() method on existing instances
=============================================================================

This module patches Conv2d.forward() at runtime on already-created instances.
This is needed because YOLOv8 creates Conv2d layers before we can patch the class.

Strategy:
---------
1. Apply patch_conv2d_forward() to replace nn.Conv2d.forward globally
2. Walk through model and replace forward() on existing instances
3. All future convolutions use im2col + matmul
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


def im2col_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
) -> torch.Tensor:
    """Pure PyTorch conv2d using im2col + matmul (bypasses MIOpen)"""
    
    # Normalize parameters to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Handle grouped convolution
    if groups == 1:
        # Step 1: Unfold input (im2col operation)
        input_unfolded = F.unfold(
            input,
            kernel_size=(kernel_h, kernel_w),
            dilation=dilation,
            padding=padding,
            stride=stride
        )
        
        # Step 2: Reshape weight for matrix multiplication
        weight_reshaped = weight.view(out_channels, -1)
        
        # Step 3: Matrix multiplication
        output = torch.matmul(weight_reshaped, input_unfolded)
        
        # Step 4: Reshape to 2D output
        output = output.view(batch_size, out_channels, out_height, out_width)
        
    else:
        # Grouped convolution - process each group separately
        channels_per_group_in = in_channels // groups
        channels_per_group_out = out_channels // groups
        
        outputs = []
        for g in range(groups):
            input_g = input[:, g * channels_per_group_in:(g + 1) * channels_per_group_in, :, :]
            weight_g = weight[g * channels_per_group_out:(g + 1) * channels_per_group_out, :, :, :]
            
            input_unfolded_g = F.unfold(
                input_g,
                kernel_size=(kernel_h, kernel_w),
                dilation=dilation,
                padding=padding,
                stride=stride
            )
            
            weight_reshaped_g = weight_g.view(channels_per_group_out, -1)
            output_g = torch.matmul(weight_reshaped_g, input_unfolded_g)
            output_g = output_g.view(batch_size, channels_per_group_out, out_height, out_width)
            
            outputs.append(output_g)
        
        output = torch.cat(outputs, dim=1)
    
    # Add bias if provided
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    
    return output


def fallback_forward(self, input):
    """Replacement forward method using im2col"""
    
    # Handle padding modes
    if self.padding_mode != 'zeros':
        input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        padding = (0, 0)
    else:
        padding = self.padding
    
    return im2col_conv2d(
        input,
        self.weight,
        self.bias,
        self.stride,
        padding,
        self.dilation,
        self.groups
    )


def patch_model_conv_layers(model):
    """
    Walk through model and patch all Conv2d forward methods
    
    This recursively finds all nn.Conv2d instances in the model
    and replaces their forward() method with our fallback.
    
    Args:
        model: PyTorch model (e.g., YOLOv8)
        
    Returns:
        Number of Conv2d layers patched
    """
    
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Replace forward method on this specific instance
            module.forward = fallback_forward.__get__(module, nn.Conv2d)
            count += 1
            print(f"  [PATCH] {name}: {module.in_channels}→{module.out_channels}, "
                  f"kernel={module.kernel_size}, groups={module.groups}")
    
    return count


def patch_all_conv2d():
    """
    Global patch: Replace Conv2d.forward() for ALL future instances
    
    This changes the class method, so all Conv2d layers created
    after this call will use the fallback implementation.
    """
    
    nn.Conv2d.forward = fallback_forward
    print("[GLOBAL PATCH] Conv2d.forward() replaced with im2col implementation")


def apply_full_patch(model=None):
    """
    Complete patching strategy:
    1. Patch the class for future instances
    2. Patch existing instances in model (if provided)
    
    Args:
        model: Optional PyTorch model to patch existing Conv2d layers
        
    Returns:
        Number of layers patched (if model provided)
    """
    
    print("="*70)
    print("APPLYING CONV2D PATCH - MIOpen Bypass")
    print("="*70)
    
    # Global patch for future instances
    patch_all_conv2d()
    
    # Patch existing instances if model provided
    if model is not None:
        print("\nPatching existing Conv2d layers in model:")
        count = patch_model_conv_layers(model)
        print(f"\n✓ Patched {count} Conv2d layers")
        print("="*70 + "\n")
        return count
    else:
        print("="*70 + "\n")
        return 0


# Convenience test
def test_patch():
    """Quick test to verify patching works"""
    
    print("Testing Conv2d patch...")
    
    # Create a conv layer
    conv = nn.Conv2d(3, 16, 3, padding=1)
    
    # Patch it
    conv.forward = fallback_forward.__get__(conv, nn.Conv2d)
    
    # Test on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conv = conv.to(device)
    
    x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    
    try:
        # Forward
        y = conv(x)
        print(f"✓ Forward: {x.shape} → {y.shape}")
        
        # Backward
        y.sum().backward()
        print(f"✓ Backward: gradients computed")
        
        print("✓ Patch test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Patch test failed: {e}")
        return False


if __name__ == "__main__":
    test_patch()
