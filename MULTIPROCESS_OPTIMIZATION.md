# DataLoader Multiprocessing Optimization - November 14, 2025

## 🎉 SUCCESS: 24% Performance Improvement!

### Results
- **workers=0**: 2.1 it/s
- **workers=4 (spawn + persistent)**: 2.6 it/s
- **Improvement**: 24% faster
- **Completion time**: 2 days earlier (Nov 24 vs Nov 26)

## Problem: Fork vs Spawn

### Why Fork Failed
The default `fork()` multiprocessing method caused deadlocks:

```
fork() → copies parent process memory
       → includes ROCm CUDA context
       → workers inherit GPU memory mappings
       → ROCm conflicts: "amdgpu_amdkfd_restore_userptr_worker hogged CPU"
       → DEADLOCK at random batch numbers
```

### Why Spawn Works
The `spawn()` method creates fresh processes:

```
spawn() → launches new Python interpreter
        → fresh CUDA context per worker
        → no shared GPU memory mappings
        → workers safely load data in parallel
        → NO DEADLOCK ✅
```

## Implementation

### 1. Force Spawn Method (CRITICAL - Must be first!)

```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # BEFORE importing torch!
```

**Why this order matters:**
- PyTorch initializes CUDA context on import
- Must set spawn method BEFORE any CUDA initialization
- Otherwise, fork context gets locked in

### 2. Monkey-Patch DataLoader

```python
original_dataloader_init = torch.utils.data.DataLoader.__init__

def patched_dataloader_init(self, *args, **kwargs):
    if kwargs.get('num_workers', 0) > 0:
        kwargs['multiprocessing_context'] = 'spawn'
        kwargs['persistent_workers'] = True
    return original_dataloader_init(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = patched_dataloader_init
```

### 3. Use workers=4 in Training

```python
results = model.train(
    workers=4,  # Now safe with spawn context!
    # ... other params
)
```

## Benefits

### 1. Spawn Context
- ✅ No fork() memory conflicts
- ✅ Fresh CUDA context per worker
- ✅ Stable with ROCm
- ✅ No deadlocks

### 2. Persistent Workers
- ✅ Workers stay alive between epochs
- ✅ Reduced process creation overhead
- ✅ Faster epoch transitions
- ✅ Lower CPU usage

### 3. Parallel Data Loading
- ✅ 4 workers load images simultaneously
- ✅ GPU doesn't wait for data
- ✅ 24% speed improvement
- ✅ Better hardware utilization

## Performance Comparison

| Configuration | Speed | Epoch Time | Total Time | Notes |
|--------------|-------|------------|------------|-------|
| workers=0 | 2.1 it/s | 5.4 hours | 11.3 days | Safe but slow |
| workers=4 (fork) | - | - | HANGS | Deadlock |
| workers=4 (spawn) | 2.6 it/s | 4.4 hours | 9.2 days | **OPTIMAL** ✅ |

## Timeline Impact

- **Original estimate** (workers=0): Nov 26 at 6:13 AM
- **Optimized** (workers=4 spawn): Nov 24 at 2:00 AM
- **Improvement**: **2 days faster!**
- **Buffer before deadline**: 6 days, 21 hours

## Key Learnings

### Initial Assumption (WRONG ❌)
"We're GPU-bound, not data-loading-bound, so workers won't help"

### Reality (CORRECT ✅)
- Data loading WAS a bottleneck
- Workers DO help (24% improvement)
- The issue was fork() conflicts, not multiprocessing itself
- spawn() context solves the conflict

### Critical Success Factors
1. Set spawn method BEFORE importing torch
2. Monkey-patch DataLoader to add spawn context
3. Enable persistent_workers for efficiency
4. Use PicklableConvForward for checkpoint saving

## Debugging Tips

If training hangs with workers>0:

1. Check multiprocessing context:
   ```python
   import multiprocessing as mp
   print(mp.get_start_method())  # Should be 'spawn'
   ```

2. Check kernel logs:
   ```bash
   sudo dmesg | tail -20 | grep -i amd
   ```

3. Verify DataLoader patch:
   ```python
   # Add debug prints in patched_dataloader_init
   print(f"Workers: {kwargs.get('num_workers')}")
   print(f"Context: {kwargs.get('multiprocessing_context')}")
   ```

## Conclusion

**Multiprocessing DOES work with ROCm + custom Conv2d patches!**

The key is using `spawn` instead of `fork` to avoid memory mapping conflicts.

Result: 24% faster training, 2 days sooner completion, stable operation. 🚀

---
**Status**: Training running optimally since Nov 14, 10:05 PM
