# ROCm DataLoader Multiprocessing Research

## Date: November 15, 2025

## Problem Summary
Training hangs with workers > 0 on AMD RX 5600 XT (gfx1010) with ROCm, even with spawn context.

## Research Findings

### 1. ROCm-Specific Issues (PyTorch #91895)
- **ROCm CI requires DataLoader tests to run SERIALLY**
- Tests hang when run in parallel with other processes
- Indicates fundamental ROCm incompatibility with multiprocessing

### 2. Tensor Clone + Multiprocessing Issue (PyTorch #78924)
- Cloning tensors BEFORE multiprocessing starts causes hangs
- Affects M1 Mac, but same pattern seen with ROCm
- **Solution**: Use `spawn` context (we already do this)
- **Root cause**: Parallel work started before fork/spawn

### 3. Shared Memory Issues
- ROCm has different shared memory behavior than CUDA
- Worker processes can't properly share GPU memory mappings
- **Workaround**: Disable shared memory with `shared_memory=False`

### 4. Thread Contention
- High core count CPUs + ROCm causes thread scheduling issues
- **Solution**: Limit threads with `torch.set_num_threads()`
- **Solution**: Set `OMP_NUM_THREADS` environment variable

### 5. Persistent Workers Issue
- `persistent_workers=True` can cause gradual resource leaks
- Workers never properly release ROCm resources
- **Solution**: Use `persistent_workers=False`

## Potential Solutions to Test

### Solution 1: Disable Shared Memory
```python
DataLoader(..., shared_memory=False)
```

### Solution 2: Limit Thread Count
```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
```

### Solution 3: Non-Persistent Workers
```python
DataLoader(..., persistent_workers=False)
```

### Solution 4: Prefetch Factor = 1
```python
DataLoader(..., prefetch_factor=1)  # Minimize queued batches
```

### Solution 5: Lower Worker Count
```python
DataLoader(..., num_workers=2)  # Instead of 4
```

### Solution 6: File System Sharing Strategy
```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

## Test Strategy

Create `train_optimized_v7_rocm_fixes.py` with ALL fixes applied:
1. `multiprocessing_context='spawn'` ✓ (already using)
2. `persistent_workers=False` (NEW)
3. `shared_memory=False` (NEW - non-standard param)
4. `prefetch_factor=2` (reduce from default 4)
5. `num_workers=2` (reduce from 4)
6. `torch.set_num_threads(2)`
7. `OMP_NUM_THREADS=2`
8. `torch.multiprocessing.set_sharing_strategy('file_system')`

If this works for 1+ epochs, gradually increase workers to find stable max.

## Fallback
If all fixes fail: workers=0 is proven stable, 19% slower but completes before deadline.
