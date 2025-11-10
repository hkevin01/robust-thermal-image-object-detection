# Automated Training Monitoring Guide

## 📊 Overview

Two powerful automation scripts have been created to monitor your training:

1. **`monitor_training_alerts.sh`** - Automated milestone alerts  
2. **`analyze_training_results.sh`** - Detailed metrics analysis

---

## 🔔 Automated Alert System

### What It Does

The monitor runs continuously in the background and will automatically alert you when:

- ✅ **Epoch 1 completes** (first epoch finished, speed will increase)
- ✅ **Epoch 10 completes** (10% done, initial assessment)
- ✅ **Epoch 30 completes** (30% done, training well established)
- ✅ **Epoch 50 completes** (halfway point, mid-training evaluation)
- ✅ **Epoch 100 completes** (training finished!)

### How It Works

**Alerts include**:
- 🖥️ Desktop notifications (pop-up messages)
- �� System sound (if available)
- 📝 Log file (`training_alerts.log`)
- 📊 Automatic analysis generation

**Current Status**:
- ✅ Monitor is running in background (PID: 413424)
- 📝 Log file: `monitor.log`
- 🔍 Check: `tail -f monitor.log`

### Manual Control

```bash
# Check if monitor is running
ps aux | grep monitor_training | grep -v grep

# View monitor log
tail -f monitor.log

# View alert history
cat training_alerts.log

# Stop monitoring (if needed)
kill $(cat .monitor_pid)

# Restart monitoring
nohup ./monitor_training_alerts.sh > monitor.log 2>&1 &
echo $! > .monitor_pid
```

---

## �� Detailed Analysis Script

### What It Provides

The analysis script (`analyze_training_results.sh`) generates comprehensive reports including:

**Training Metrics**:
- ✅ Latest epoch losses (box, class, DFL)
- ✅ Best performance metrics (highest mAP@0.5)
- ✅ Loss trends (last 10 epochs)
- ✅ mAP trends (precision, recall, mAP@0.5, mAP@0.5:0.95)

**Performance Assessment**:
- ✅ Current training quality (Excellent/Good/Fair/Starting)
- ✅ Progress percentage
- ✅ Estimated time remaining

**System Health**:
- ✅ Error detection (HSA errors, general errors)
- ✅ Process status (CPU, memory usage)
- ✅ Output files status

**Recommendations**:
- ✅ Actionable next steps based on current progress
- ✅ Performance optimization suggestions
- ✅ When to stop or continue training

### Usage

```bash
# Run manual analysis
./analyze_training_results.sh

# Save analysis to file
./analyze_training_results.sh > my_analysis.txt

# Watch analysis in real-time
watch -n 60 ./analyze_training_results.sh
```

### Automatic Analysis

The monitor automatically runs analysis and saves reports at each milestone:

- `analysis_epoch_1.txt` - After first epoch
- `analysis_epoch_10.txt` - At 10% complete
- `analysis_epoch_30.txt` - At 30% complete
- `analysis_epoch_50.txt` - At midpoint
- `analysis_epoch_100_FINAL.txt` - Final comprehensive report

---

## 📋 Quick Status Checks

### One-Line Status
```bash
# Quick progress check
tail -100 training.log | grep -E "[0-9]+/100" | tail -1

# Completed epochs
grep -c "Validating:" training.log

# Current losses
tail -1 runs/detect/production_yolov8n_rocm522/results.csv 2>/dev/null || echo "Epoch 1 not complete yet"
```

### Process Health
```bash
# Is training running?
ps aux | grep 407777 | grep -v grep && echo "✅ Training active" || echo "❌ Training stopped"

# Is monitor running?
ps aux | grep monitor_training | grep -v grep && echo "✅ Monitor active" || echo "❌ Monitor stopped"

# Resource usage
ps aux | grep 407777 | awk '{printf "CPU: %s%% | RAM: %.1f GB\n", $3, $6/1024/1024}'
```

### Expected Timeline Reminders

**First Epoch** (Currently in progress):
- Duration: ~3-4 hours
- Status: Slow (MIOpen kernel compilation)
- Check: Around 22:30-23:00 tonight

**Epochs 2-10** (Tomorrow morning):
- Duration: ~3-4 hours total (15-25 min each)
- Status: Fast (using cached kernels)
- Check: Tomorrow ~8:00 AM

**Epoch 50** (Midpoint, ~24h from start):
- Check: Tomorrow evening ~18:47
- Target: mAP@0.5 > 0.5

**Epoch 100** (Complete, ~48-72h):
- Check: Nov 11-12
- Target: mAP@0.5 > 0.7

---

## 🎯 What to Expect

### Notification Timeline

**Tonight (~22:30-23:00)**:
```
🔔 ALERT
   ✅ First Epoch Complete!
   Epoch 1/100 finished in 3:40:00. Training speed will now accelerate!

🔔 ALERT  
   📊 Analysis Generated
   Detailed analysis saved to analysis_epoch_1.txt
```

**Tomorrow Morning (~8:00 AM)**:
```
🔔 ALERT
   🎯 Checkpoint: 10 Epochs
   10% complete! Review analysis_epoch_10.txt for metrics.
```

**Tomorrow Evening (~18:47)**:
```
🔔 ALERT
   🏁 Halfway Point!
   50 epochs complete! Check mAP50 metrics in analysis_epoch_50.txt
```

**Nov 11-12 (Completion)**:
```
🔔 ALERT
   🎉 TRAINING COMPLETE!
   All 100 epochs finished! Final model ready for validation.
   
🔔 ALERT
   📊 Final Analysis
   Complete training analysis saved to analysis_epoch_100_FINAL.txt
```

---

## 📁 Generated Files

### Automatic Logs
- `monitor.log` - Monitor script output
- `training_alerts.log` - All alerts history
- `.training_state` - Monitor state (tracks which alerts sent)
- `.monitor_pid` - Monitor process ID

### Analysis Reports
- `analysis_epoch_1.txt` - First epoch analysis
- `analysis_epoch_10.txt` - 10% checkpoint
- `analysis_epoch_30.txt` - 30% checkpoint
- `analysis_epoch_50.txt` - Midpoint analysis
- `analysis_epoch_100_FINAL.txt` - Complete final report

### Training Outputs (from YOLOv8)
- `training.log` - Main training log
- `runs/detect/production_yolov8n_rocm522/results.csv` - Epoch metrics
- `runs/detect/production_yolov8n_rocm522/results.png` - Training curves
- `runs/detect/production_yolov8n_rocm522/weights/best.pt` - Best model
- `runs/detect/production_yolov8n_rocm522/weights/last.pt` - Latest checkpoint

---

## 🔧 Troubleshooting

### Monitor Not Alerting

```bash
# Check if monitor is running
ps aux | grep monitor_training

# Check monitor log for errors
tail -50 monitor.log

# Restart monitor
kill $(cat .monitor_pid)
rm -f .training_state monitor.log
nohup ./monitor_training_alerts.sh > monitor.log 2>&1 &
echo $! > .monitor_pid
```

### Analysis Script Errors

```bash
# Check if results.csv exists
ls -lh runs/detect/production_yolov8n_rocm522/results.csv

# If not, epoch 1 hasn't completed yet
# The analysis will work once epoch 1 finishes
```

### Desktop Notifications Not Showing

Desktop notifications require `notify-send` (usually installed by default on Ubuntu). If not showing:

```bash
# Install notification daemon
sudo apt install libnotify-bin

# Test notification
notify-send "Test" "This is a test notification"
```

---

## 💡 Pro Tips

### Silent Monitoring (No Terminal Needed)

The monitor runs in background - you can close the terminal!

```bash
# Check status anytime from new terminal
cd /home/kevin/Projects/robust-thermal-image-object-detection
tail -20 training_alerts.log

# Quick progress
./check_training_progress.sh
```

### Email/SMS Alerts (Advanced)

If you want email alerts when milestones complete, add to `monitor_training_alerts.sh` alert function:

```bash
# Add to alert() function
echo "$message" | mail -s "$title" your@email.com
```

### Remote Monitoring

Check from another computer:

```bash
ssh kevin@your-server
cd /home/kevin/Projects/robust-thermal-image-object-detection
./check_training_progress.sh
```

---

## 📞 Next Steps

### Right Now
- ✅ Monitoring is active and running
- ✅ Training progressing through first epoch
- ✅ No action needed - let it run!

### When You Get First Alert (~Tonight)
1. Read `analysis_epoch_1.txt`
2. Verify losses decreased
3. Check mAP50 starting values
4. Confirm no errors

### At Epoch 50 (Tomorrow Evening)
1. Review `analysis_epoch_50.txt`
2. Check if mAP@0.5 > 0.5 (target)
3. Evaluate training curves in `results.png`
4. Decide: continue to 100 or stop early

### At Completion (Nov 11-12)
1. Read `analysis_epoch_100_FINAL.txt`
2. Validate model: `yolo detect val model=runs/detect/production_yolov8n_rocm522/weights/best.pt data=data/ltdv2_full/data.yaml`
3. Test predictions: `yolo detect predict model=runs/detect/production_yolov8n_rocm522/weights/best.pt source=data/ltdv2_full/images/test`
4. Document results in `ROCM_52_TRAINING_SUCCESS.md`
5. Create backup: `tar -czf training_backup_$(date +%Y%m%d).tar.gz runs/detect/production_yolov8n_rocm522 *.log *.txt`

---

## 🎉 You're All Set!

The automation is handling everything:
- ✅ Continuous monitoring
- ✅ Milestone alerts  
- ✅ Detailed analysis
- ✅ Progress tracking

**Just let it run and wait for the alerts!**

---

*Last Updated: November 9, 2025, 19:05*  
*Training PID: 407777*  
*Monitor PID: 413424*  
*Status: 🟢 Both running successfully*
