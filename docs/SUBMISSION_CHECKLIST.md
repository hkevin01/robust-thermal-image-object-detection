# Competition Submission Checklist

## ✅ Submission Preparation - Complete Guide

### Current Status: Training Phase

**Training Started**: Nov 10, 2025 10:33 AM  
**Current Time**: ~76 minutes running  
**Progress**: Epoch 1/50 initialization  
**Expected Completion**: ~Nov 18, 2025  

---

## 📋 Phase 1: Pre-Submission (NOW - Training Completes)

### Training Monitoring
- [x] Training script launched successfully
- [ ] Training progressing normally (check every 6 hours)
- [ ] GPU temperatures stable (< 85°C)
- [ ] Loss curves decreasing
- [ ] No crashes or errors

**Monitor Commands**:
```bash
# Check if training is running
ps aux | grep train_patched.py | grep -v grep

# View latest progress
tail -50 logs/training_production.log

# Watch training live
tail -f logs/training_production.log
```

### Documentation Ready
- [x] Submission format documented
- [x] Class mapping clarified (4 classes, not 5)
- [x] Validation scripts created
- [x] Generation scripts created
- [x] Common issues documented
- [x] data.yaml corrected (nc: 4)

---

## �� Phase 2: Post-Training (After ~9 Days)

### Step 1: Verify Training Results

Check training completed successfully:
```bash
# Check if training finished
ls -lh runs/detect/train*/weights/best.pt

# View final metrics
cat runs/detect/train*/results.csv | tail -5

# Check validation mAP
grep "mAP" runs/detect/train*/results.csv | tail -1
```

**Expected Results**:
- ✓ best.pt weights file exists
- ✓ last.pt weights file exists  
- ✓ results.csv shows 50 epochs
- ✓ Validation mAP@0.5 > 0.30 (baseline target)

### Step 2: Generate Development Submission

Generate predictions on validation set:
```bash
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/valid/images \
    --output submission_dev_v1.json \
    --conf 0.001 \
    --device 0
```

**Expected Output**:
- JSON file with ~100K-200K predictions
- Average 2-5 detections per image (41,226 images)
- File size: 10-50MB
- Per-class distribution: Person (most), Vehicle (many), Bicycle/Motorcycle (fewer)

### Step 3: Validate Submission

Run validation:
```bash
python scripts/validate_submission.py submission_dev_v1.json
```

**Must Pass**:
- ✓ Valid JSON format
- ✓ All required fields present
- ✓ Category IDs in [1, 2, 3, 4]
- ✓ Bbox format correct [x, y, w, h]
- ✓ Scores in [0.0, 1.0]
- ✓ All validation images covered

**If Validation Fails**:
1. Read error messages carefully
2. Check `docs/COMPETITION_SUBMISSION_GUIDE.md` for fixes
3. Fix issues and regenerate
4. Validate again

### Step 4: Upload to Codabench

1. Go to: https://www.codabench.org/competitions/10954/
2. Sign in / Register (if needed)
3. Click "Participate" tab
4. Select "Development Phase"
5. Click "Submit" button
6. Upload `submission_dev_v1.json`
7. Add submission description: "YOLOv8n baseline, 50 epochs"
8. Click "Submit"

**Wait for Evaluation**:
- Processing time: 10-30 minutes
- You'll receive email when done
- Check "Results" tab for leaderboard

### Step 5: Analyze Results

Once scored, analyze:
```bash
# Create results directory
mkdir -p analysis/submission_v1

# Download leaderboard CSV (if available)
# Save per-month scores (if provided)
# Document findings
```

**Key Metrics to Record**:
- Overall mAP@0.5: _______
- Per-month mAPs (if shown):
  - June 2021: _______
  - July 2021: _______
  - August 2021: _______
  - September 2021: _______
  - October 2021: _______
- CoV: _______
- Robustness Score: _______
- Leaderboard Rank: _____ / 32

**Questions to Answer**:
1. Which months performed worst?
2. Which classes have lowest AP?
3. Are we competitive with top teams?
4. What's the gap to #1?

---

## 📋 Phase 3: Iteration (Development Phase)

### Improvement Strategies

Based on analysis, try:

**Strategy A: Longer Training**
```bash
# Train for 100 epochs
python train_patched.py --epochs 100
```

**Strategy B: Different Model Size**
```bash
# Try YOLOv8s (larger, more accurate)
python train_patched.py --model yolov8s.pt
```

**Strategy C: Confidence Threshold Tuning**
```bash
# Try different thresholds
for conf in 0.001 0.01 0.05 0.1; do
    python scripts/generate_submission.py \
        --model runs/detect/train/weights/best.pt \
        --test-dir data/ltdv2_full/valid/images \
        --output submission_dev_conf${conf}.json \
        --conf $conf
    
    python scripts/validate_submission.py submission_dev_conf${conf}.json
    # Upload and compare scores
done
```

**Strategy D: Test-Time Augmentation**
- Modify generation script to run multiple scales
- Average predictions
- Apply NMS

**Strategy E: Ensemble**
- Train 3-5 models with different seeds/configs
- Merge predictions
- Weight by validation mAP

### Iteration Checklist

For each improvement:
- [ ] Train/configure model
- [ ] Generate submission
- [ ] Validate format
- [ ] Upload to Codabench
- [ ] Record score
- [ ] Compare to previous best
- [ ] Keep if better, discard if worse

---

## 📋 Phase 4: Final Submission (Dec 1-7)

### Pre-Final Checks

Before submitting on test set:
- [ ] Best model selected from development phase
- [ ] Confidence threshold optimized
- [ ] Submission pipeline tested multiple times
- [ ] Model weights backed up
- [ ] Code archived

### Generate Final Submission

**CRITICAL**: Test set, not validation set!

```bash
# Generate predictions on TEST set
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/test/images \
    --output submission_final.json \
    --conf 0.01 \
    --device 0

# Validate
python scripts/validate_submission.py submission_final.json

# Backup
cp submission_final.json submission_final_backup_$(date +%Y%m%d_%H%M%S).json
```

**Expected**:
- ~150K predictions (46,884 test images)
- File size: 15-60MB
- Same validation checks pass

### Upload Final Submission

1. Go to: https://www.codabench.org/competitions/10954/
2. Click "Participate" tab
3. Select "**Final Testing Phase**" ⚠️
4. Upload `submission_final.json`
5. Description: "YOLOv8n, 50 epochs, conf=0.01"
6. Double-check phase is correct!
7. Submit

**Remember**:
- Limited submissions in final phase!
- No feedback until competition ends (Dec 7)
- Choose wisely!

### Post-Submission

- [ ] Submission uploaded successfully
- [ ] Confirmation email received
- [ ] Submission ID recorded: __________
- [ ] Awaiting final results (Dec 7)

---

## 📋 Phase 5: Paper Submission (Optional, Dec 14)

If top 3 or interesting approach:
- [ ] Write competition paper (4-8 pages)
- [ ] Include: Method, experiments, results, analysis
- [ ] Follow WACV format
- [ ] Submit by Dec 14
- [ ] Attend workshop (Jan 2026)

---

## 🚨 Critical Reminders

### Do NOT:
- ❌ Submit on test set during development phase
- ❌ Use test set for validation or tuning
- ❌ Share model weights publicly before competition ends
- ❌ Use external thermal datasets (check rules!)

### DO:
- ✅ Test submission pipeline on validation set first
- ✅ Validate every submission before upload
- ✅ Keep all model weights backed up
- ✅ Document every experiment
- ✅ Monitor competition announcements
- ✅ Ask organizers if unclear (asjo@create.aau.dk)

---

## 📞 Support Resources

- **Competition Page**: https://www.codabench.org/competitions/10954/
- **Workshop**: https://vap.aau.dk/rws/
- **Organizer Email**: asjo@create.aau.dk
- **Documentation**:
  - Submission guide: `docs/COMPETITION_SUBMISSION_GUIDE.md`
  - Workflow: `docs/SUBMISSION_WORKFLOW.md`
  - Architecture: `docs/COMPETITION_APPROACH.md`

---

## 📊 Success Metrics

### Minimum Target (Baseline)
- Development mAP@0.5: > 0.30
- Robustness Score: > 0.25
- Leaderboard: Top 50%

### Competitive Target
- Development mAP@0.5: > 0.45
- Robustness Score: > 0.40
- Leaderboard: Top 25%

### Stretch Goal
- Development mAP@0.5: > 0.55
- Robustness Score: > 0.50
- Leaderboard: Top 10

---

**Last Updated**: Nov 10, 2025  
**Next Review**: When training completes (~Nov 18)  
**Competition Ends**: Nov 30, 2025 (Development), Dec 7, 2025 (Final)
