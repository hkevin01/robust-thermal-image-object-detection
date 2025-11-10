# Competition Submission Workflow - Quick Reference

## �� Quick Start (Development Phase)

```bash
# 1. Wait for training to complete
# Monitor: tail -f logs/training_production.log

# 2. Generate predictions on validation set
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/valid/images \
    --output submission_dev.json \
    --conf 0.001 \
    --device 0

# 3. Validate submission
python scripts/validate_submission.py submission_dev.json

# 4. Upload to Codabench
# Go to: https://www.codabench.org/competitions/10954/
# Click "Participate" → "Submit" → Upload submission_dev.json
```

## 📊 Competition Details

| Item | Value |
|------|-------|
| Platform | Codabench |
| URL | https://www.codabench.org/competitions/10954/ |
| Current Phase | Development (ends Nov 30, 2025 6:55 PM EST) |
| Time Remaining | ~20 days |
| Participants | 32 |
| Submissions | 127 |

## 🎓 Classes (4 total)

| ID | Name | Description |
|----|------|-------------|
| 1 | Person | Pedestrians, people |
| 2 | Bicycle | Bicycles |
| 3 | Motorcycle | Motorcycles, scooters |
| 4 | Vehicle | Cars, buses, trucks |

**Note**: Competition uses category_id 1-4, YOLO uses class 0-3. Scripts handle conversion automatically.

## 📐 Submission Format

```json
[
    {
        "image_id": 1,
        "category_id": 1,
        "bbox": [x, y, width, height],
        "score": 0.95
    }
]
```

**Critical**: 
- Bbox format: `[x, y, width, height]` NOT `[x1, y1, x2, y2]`
- Coordinates: Absolute pixels, not normalized
- Scores: Float in [0.0, 1.0]

## 🔢 Evaluation Metric

$$\text{Robustness Score} = \text{mAP@0.5} \times (1 - \text{CoV}_{\text{monthly}})$$

**What this means**:
- High mAP = Good detection accuracy
- Low CoV = Consistent across months (robust to thermal drift)
- Both matter! Consistency can beat raw accuracy

## 📅 Timeline

| Date | Action |
|------|--------|
| Now | Training in progress (Epoch 1/50) |
| ~Nov 18 | Training completes (~9 days) |
| Nov 19-20 | Generate & validate submission |
| Nov 21 | First development submission |
| Nov 21-29 | Iterate based on feedback |
| Nov 30 | Final development submission |
| Dec 1 | Submit on test set (limited attempts!) |

## ⚠️ Common Pitfalls

1. **Wrong bbox format**: Use `[x, y, w, h]` not `[x1, y1, x2, y2]`
2. **Wrong class IDs**: Competition uses 1-4, not 0-3
3. **Normalized coords**: Use absolute pixels
4. **Missing images**: Process all test images (even if no detections)
5. **Large file**: Compress with `gzip` if > 100MB

## 🛠️ Troubleshooting

### Validation fails
```bash
# Check what's wrong
python scripts/validate_submission.py submission.json

# Common fixes are documented in the output
```

### File too large
```bash
# Check size
ls -lh submission.json

# Compress
gzip submission.json
# Upload .gz file to Codabench
```

### Wrong image IDs
```bash
# List validation image IDs
ls data/ltdv2_full/valid/images/ | head -20

# Filenames should match image_id in JSON
```

## 📚 Documentation

- **Full submission guide**: `docs/COMPETITION_SUBMISSION_GUIDE.md`
- **Architecture details**: `docs/COMPETITION_APPROACH.md`
- **Project README**: `README.md`

## 🚀 Next Steps After First Submission

1. **Check leaderboard** - Compare to other teams
2. **Analyze per-month scores** (if provided) - Find weak months
3. **Improve model**:
   - Add temporal augmentation
   - Train longer
   - Try ensemble
   - Tune confidence threshold
4. **Resubmit** - Unlimited in development phase!

## 🎯 Success Checklist

### Before Training
- [x] Data downloaded (1M+ images)
- [x] Data converted to YOLO format
- [x] Training configuration set
- [x] Training script ready (train_patched.py)

### During Training
- [x] Training launched
- [ ] Training stable and progressing
- [ ] Losses decreasing
- [ ] GPU healthy
- [ ] ETA: ~9 days

### After Training
- [ ] Best weights saved
- [ ] Validation mAP checked
- [ ] Predictions generated
- [ ] Submission validated
- [ ] Uploaded to Codabench
- [ ] Leaderboard checked

### Optimization
- [ ] First baseline submission done
- [ ] Per-month performance analyzed
- [ ] Improvements identified
- [ ] Multiple models trained
- [ ] Ensemble created
- [ ] Confidence threshold tuned
- [ ] Final submission ready

---

**Status**: Training in progress 🔄  
**Next Milestone**: Training completion (~Nov 18)  
**Competition Deadline**: Nov 30, 2025 6:55 PM EST  
