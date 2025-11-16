# Training Hang Analysis - November 15, 2025

## Timeline of Events

1. **Nov 14, 10:05 PM**: Training started with workers=4, spawn context
2. **Nov 15, 02:41 AM**: Epoch 1 completed successfully (4.6 hours)
3. **Nov 15, 02:41 AM**: Checkpoints saved (best.pt, last.pt - 12MB each)
4. **Nov 15, 02:41 AM**: **HANG OCCURRED** - Batch 4,886 in Epoch 2
5. **Nov 15, 10:29 AM**: Discovered hung (7.8 hours stuck)

## Evidence

### What Worked
- ✅ Epoch 1 completed (41,163 batches)
- ✅ Checkpoints saved successfully (PicklableConvForward working)
- ✅ Multiprocessing stable during Epoch 1
- ✅ Speed maintained at 2.6 it/s throughout Epoch 1

### Hang Location
- **Batch**: 4,886 / 41,163 (11.9% into Epoch 2)
- **Time**: Immediately after validation/checkpoint saving
- **Duration**: ~7.8 hours stuck

### Kernel Evidence
```
workqueue: amdgpu_amdkfd_restore_userptr_worker [amdgpu] hogged CPU for >10000us
```
Same error as before with fork() - suggesting ROCm memory mapping issue persists!

## Root Cause Analysis

### Theory: Validation Phase Triggers Hang

The hang occurred RIGHT AFTER epoch 1 validation, not during training. Possible causes:

1. **Validation with spawn workers**: 
   - Validation may use different DataLoader settings
   - Workers might not be reusing spawn context
   - Fresh worker spawning during validation causes conflicts

2. **Checkpoint save triggers worker restart**:
   - After checkpoint, workers might respawn
   - Respawned workers might use fork() instead of spawn()
   - ROCm conflict on worker restart

3. **Persistent workers issue**:
   - Workers from Epoch 1 try to start Epoch 2
   - Memory mappings corrupt after validation
   - Workers hang on next batch

## Solutions to Try

### Option 1: Disable Validation (Fast Test)
```python
val=False  # Skip validation entirely
```
- Pros: Quick test, will complete if validation is the issue
- Cons: No mAP metrics, no model selection

### Option 2: Disable Persistent Workers
```python
persistent_workers=False
```
- Pros: Workers restart each epoch, fresh state
- Cons: More overhead, might be slower

### Option 3: Back to workers=0 (Safe)
```python
workers=0  # Single-threaded
```
- Pros: Guaranteed to work
- Cons: 24% slower (but still completes on time)

### Option 4: Reduce workers
```python
workers=2  # Fewer workers = less contention
```
- Pros: Some parallelism, less complexity
- Cons: Still might hang

## Recommendation

**Try Option 1 first**: Disable validation to isolate if that's the trigger.
- If it completes without hanging → validation is the issue
- Can add back validation every N epochs (e.g., every 5 epochs)
- Still get checkpoints, just no validation metrics

**Fallback to Option 3**: workers=0 if validation isn't the issue.
- Guaranteed completion
- Still meets deadline (Nov 26 vs Nov 24)

## Next Steps

1. ✅ Kill hung process
2. ✅ Create improved training script with:
   - Better logging
   - val=False for testing
   - Resume from last.pt checkpoint
3. Start training and monitor closely
4. If completes, gradually re-enable features

