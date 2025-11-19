# Quick Reference - Training Monitor

**Training Started**: Nov 12, 2025 10:47 AM EST  
**Expected Completion**: Nov 15, 2025 1:47 AM EST (62.6 hours)  
**Script**: `train_optimized_v2.py` ✅ WORKING

---

## 📊 Check Status (Right Now)

```bash
# Is training running?
ps aux | grep train_optimized_v2

# GPU status
rocm-smi --showuse --showtemp

# Latest progress
tail -20 logs/training_optimized_v2_*.log | grep -E "Epoch|batch"

# Checkpoints
ls -lh runs/detect/train_optimized_v2/weights/
```

---

## 📈 What to Expect

**Good Training**:
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
 1/50    2.1G     0.987     0.845     0.912         42    640: 12% ━━━━  1000/82325 ~18it/s
```

**Bad Training**:
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
 1/50    2.1G       nan       nan       nan         42    640: 0% ━    50/82325 0.0it/s
```

---

## ⚡ Quick Actions

**Monitor Live**:
```bash
tail -f logs/training_optimized_v2_*.log
```

**Check GPU**:
```bash
watch -n 1 rocm-smi --showuse --showtemp
```

**View Checkpoints**:
```bash
ls -lth runs/detect/train_optimized_v2/weights/
```

**Resume If Stopped**:
```bash
cd ~/Projects/robust-thermal-image-object-detection
source venv-py310-rocm52/bin/activate
python train_optimized_v2.py
```

---

## 🎯 Success Metrics

- ✅ **Speed**: ~18 batches/sec
- ✅ **GPU**: 80-95% usage
- ✅ **Temp**: 50-70°C
- ✅ **Memory**: ~2.0-2.5 GB
- ✅ **Losses**: Numbers (not NaN), decreasing

---

##  📁 Key Files

```
train_optimized_v2.py          ← Main script
patches/conv2d_optimized.py    ← Conv2d fallback
docs/TRAINING_SUCCESS_NOV12.md ← Full documentation
logs/training_optimized_v2_*.log ← Current log
runs/detect/train_optimized_v2/ ← Checkpoints & results
```

---

## 🆘 Troubleshooting

**NaN losses**:
- Check log for errors before NaN appears
- May need to lower learning rate (lr0: 0.001)

**Slow speed (<10 batch/s)**:
- Check GPU usage (should be 80-95%)
- Check temperature (may be thermal throttling)
- Check background processes

**Frozentraining (no progress >5 min)**:
- Check `ps aux | grep python`
- Check dmesg: `sudo dmesg | tail -50`
- May need to restart

---

## 📅 Timeline

| Date | Progress | Action |
|------|----------|--------|
| Nov 12 | 0% | ✅ Training started |
| Nov 13 | 38% | Monitor stability |
| Nov 14 | 76% | Verify checkpoints |
| Nov 15 | 100% | Evaluate results |
| Nov 30 | - | Submit to Codabench |

---

**💪 We've got this. The hard part is done.**
