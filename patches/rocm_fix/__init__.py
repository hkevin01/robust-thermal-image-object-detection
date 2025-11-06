"""
ROCm RDNA1/2 Memory Coherency Fix Package
==========================================

Addresses: Memory access fault by GPU (Page not present or supervisor privilege)
Target: AMD RX 5600 XT (gfx1010), RX 6000 series (gfx1030)
ROCm versions: 6.2+

Usage:
    from patches.rocm_fix import apply_rocm_fix
    apply_rocm_fix()
"""

from .hip_memory_patch import apply_rocm_fix, test_allocation

__all__ = ['apply_rocm_fix', 'test_allocation']
__version__ = '1.0.0'
