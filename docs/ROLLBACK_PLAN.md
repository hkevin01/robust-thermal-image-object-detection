# Rollback Plan: Redo Epoch 3 with NaN Prevention

## Current Situation

**Problem:** Epoch 3 had 2 NaN incidents (batches 15 & 46) during previous training run
**Current State:** Training resumed from Epoch 3 → 4 with NaN prevention active
**Checkpoint Status:** 
- `last.pt` = Epoch 1 (before NaN)
- `best.pt` = Epoch 1 (before NaN)

**Issue:** The Epoch 3 checkpoint that was used to resume contains NaN-corrupted training history.

## Strategy

### Option 1: Let Current Training Complete Epoch 4, Then Rollback ✅ RECOMMENDED

**Reasoning:**
- Epoch 4 is running with **full NaN prevention** (gradient clipping, detection, etc.)
- We can verify NaN prevention works on a complete epoch
- Then rollback to clean Epoch 1 checkpoint and redo Epochs 2-4 properly

**Steps:**
1. ✅ **Wait for Epoch 4 to complete** (~5 hours)
2. ✅ **Verify Epoch 4 had zero NaN** (validation checkpoint)
3. ✅ **Stop training**
4. ✅ **Restore from Epoch 1 checkpoint** (last.pt = clean)
5. ✅ **Resume training from Epoch 2** with NaN prevention
6. ✅ **Complete Epochs 2-50** cleanly

### Option 2: Stop Now and Rollback Immediately

**Reasoning:**
- Don't waste time on Epoch 4 built on NaN-corrupted Epoch 3
- Start fresh immediately

**Steps:**
1. Stop training now
2. Restore Epoch 1 checkpoint
3. Resume from Epoch 2 with NaN prevention
4. Complete Epochs 2-50 cleanly

## Recommendation: **Option 1**

### Why Option 1 is Better:

1. **Validation of NaN Prevention**
   - Epoch 4 is our first complete epoch with all protections
   - We need to verify it works before committing to 46 more epochs
   - Only ~5 hours to get proof of concept

2. **Risk Mitigation**
   - If Epoch 4 still gets NaN, we know we need stronger measures
   - Better to find out now than after redoing Epochs 2-3

3. **Minimal Time Loss**
   - Epoch 4: ~5 hours
   - Rollback + Redo 2-3: ~10 hours
   - Total: ~15 hours
   - vs. Immediate rollback: ~10 hours (but no validation)

4. **Training Metrics**
   - Epoch 4 will show us the impact of:
     - Reduced learning rate
     - Gradient clipping effectiveness
     - Loss curve smoothness

## Implementation Plan (Option 1)

### Phase 1: Monitor Epoch 4 ⏳ IN PROGRESS

```bash
# Watch for NaN
tail -f logs/training_NAN_PREVENTION_20251117.log | grep -E "nan|NaN|WARNING|Epoch.*box_loss"

# Count NaN (should stay at 0)
watch -n 60 'grep -c " nan " logs/training_NAN_PREVENTION_20251117.log'
```

**Success Criteria:**
- [ ] Epoch 4 completes all 41,163 batches
- [ ] Zero NaN occurrences
- [ ] Validation completes successfully
- [ ] Loss curves are smooth

### Phase 2: Stop and Rollback (After Epoch 4)

```bash
# 1. Stop training gracefully
pkill -SIGTERM -f train_v7_final_working.py
sleep 5

# 2. Backup current state
cp runs/detect/train_optimized_v5_multiprocess/weights/last.pt \
   runs/detect/train_optimized_v5_multiprocess/weights/last_epoch3_nan.pt.backup

# 3. Verify Epoch 1 checkpoint is clean
./venv/bin/python << PYEOF
import torch
ckpt = torch.load('runs/detect/train_optimized_v5_multiprocess/weights/best.pt', map_location='cpu')
print(f"Epoch: {ckpt['epoch']}")
print(f"Fitness: {ckpt['best_fitness']}")
PYEOF

# 4. Copy clean Epoch 1 checkpoint to last.pt
cp runs/detect/train_optimized_v5_multiprocess/weights/best.pt \
   runs/detect/train_optimized_v5_multiprocess/weights/last.pt

# 5. Update checkpoint to resume from Epoch 2
./venv/bin/python << PYEOF
import torch
ckpt = torch.load('runs/detect/train_optimized_v5_multiprocess/weights/last.pt', map_location='cpu')
ckpt['epoch'] = 1  # Will resume as Epoch 2
torch.save(ckpt, 'runs/detect/train_optimized_v5_multiprocess/weights/last.pt')
print("✅ Checkpoint ready to resume from Epoch 2")
PYEOF
```

### Phase 3: Resume Clean Training

```bash
# Start training from clean Epoch 2 with NaN prevention
nohup ./venv/bin/python -u train_v7_final_working.py \
  > logs/training_CLEAN_RESTART_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > training.pid

# Monitor
tail -f logs/training_CLEAN_RESTART_*.log
```

**Expected Results:**
- Epochs 2-50 complete with zero NaN
- Clean training history
- Better final mAP (no NaN corruption)

## Alternative: Option 2 Commands (Stop Now)

If you want to stop immediately:

```bash
# Stop training
pkill -SIGTERM -f train_v7_final_working.py

# Follow Phase 2 & 3 above
```

## Decision Point

**When to execute:** 
- **Option 1:** After Epoch 4 completes (~5 hours from now)
- **Option 2:** Immediately

**Current recommendation:** Wait for Epoch 4 to validate NaN prevention works

## Monitoring Dashboard

```bash
# Check current epoch progress
tail -20 logs/training_NAN_PREVENTION_20251117.log | grep -E "Epoch|box_loss"

# NaN count (should stay 0 for Epoch 4)
echo "NaN count in current log: $(grep -c ' nan ' logs/training_NAN_PREVENTION_20251117.log)"

# Process status
ps -p $(cat training.pid) -o pid,etime,pcpu,pmem,cmd
```

## Timeline

```
Now:           Epoch 4 in progress (with NaN prevention)
+5 hours:      Epoch 4 completes
+5h:           Stop, rollback to Epoch 1
+5h 10min:     Resume from Epoch 2 (clean)
+10h:          Epoch 2 completes (clean)
+15h:          Epoch 3 completes (clean, no NaN!)
+20h:          Epoch 4 completes again (clean)
...
+240h (~10d):  Epoch 50 completes
```

## Risk Assessment

**Option 1 Risks:**
- ❌ ~5 hours on potentially wasted Epoch 4
- ✅ But we validate NaN prevention first

**Option 2 Risks:**
- ✅ No time wasted
- ❌ But no validation before committing to 48 epochs

**Recommendation:** Option 1's validation is worth the 5-hour investment

---

**Status:** ⏳ Waiting for Epoch 4 to complete
**Action Required:** Review this plan and decide when to execute rollback
**Documentation:** This file + docs/NAN_PREVENTION.md
