# WACV 2026 RWS: Competition Submission Guide

## Competition Overview

- **Platform**: Codabench
- **URL**: https://www.codabench.org/competitions/10954/
- **Organized by**: Asj-Aau (asjo@create.aau.dk)
- **Workshop**: 6th Real World Surveillance Workshop (RWS) at WACV 2026
- **Docker Image**: asjaau/ltdv2:latest

### Important Dates

| Date | Event |
|------|-------|
| Oct 17, 2025 | Competition start, Development Phase begins |
| Nov 30, 2025 6:55 PM EST | Development Phase ends |
| Dec 1, 2025 | Final Testing Phase begins |
| Dec 7, 2025 | Competition ends |
| Dec 14, 2025 | Paper submission deadline |
| Dec 23, 2025 | Decision notification |
| Jan 9, 2026 | Camera-ready deadline |

### Phases

**Development Phase** (Oct 17 - Nov 30):
- Submit predictions on **validation set** (41,226 images)
- Get immediate feedback on leaderboard
- Tune models and hyperparameters
- Unlimited submissions (use wisely!)

**Final Testing Phase** (Dec 1 - Dec 7):
- Submit predictions on **test set** (46,884 images)
- Limited submissions (check competition rules)
- NO FEEDBACK until competition ends
- Final ranking determined by these scores

## Submission Format

### JSON Structure

Submissions must be in **JSON array format** (list of predictions):

```json
[
    {
        "image_id": 1,
        "category_id": 1,
        "bbox": [100.5, 200.3, 50.2, 80.1],
        "score": 0.95
    },
    {
        "image_id": 1,
        "category_id": 4,
        "bbox": [300.0, 150.0, 120.5, 90.3],
        "score": 0.88
    }
]
```

### Field Requirements

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `image_id` | int | Test image ID | Must match test set filenames |
| `category_id` | int | Object class | 1=Person, 2=Bicycle, 3=Motorcycle, 4=Vehicle |
| `bbox` | list[float] | Bounding box | Format: `[x, y, width, height]` in absolute pixels |
| `score` | float | Confidence score | Range: `[0.0, 1.0]` |

### Class Mapping

LTDv2 uses **4 classes**:

| category_id | Class Name | Description |
|-------------|------------|-------------|
| 1 | Person | Pedestrians, people |
| 2 | Bicycle | Bicycles (motorized or not) |
| 3 | Motorcycle | Motorcycles, scooters |
| 4 | Vehicle | Cars, buses, trucks |

**Note**: If training on COCO-pretrained models, map multiple COCO classes to LTDv2 Vehicle class:
- COCO car (2) → LTDv2 Vehicle (4)
- COCO bus (5) → LTDv2 Vehicle (4)
- COCO truck (7) → LTDv2 Vehicle (4)

### Bounding Box Format

**CRITICAL**: Use `[x, y, width, height]` format, NOT `[x1, y1, x2, y2]`!

```python
# Correct format
bbox = [x_topleft, y_topleft, box_width, box_height]

# Convert from x1,y1,x2,y2 if needed
x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
```

- Coordinates are in **absolute pixels**, not normalized
- All values must be `>= 0`
- Width and height must be `> 0`

## Evaluation Metric

### Robustness Score

$$\text{Robustness Score} = \text{mAP@0.5} \times (1 - \text{CoV}_{\text{monthly}})$$

Where:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (standard COCO metric)
- **CoV (Coefficient of Variation)**: $\text{CoV} = \frac{\sigma}{\mu}$ of per-month mAP scores

### Why This Metric?

The competition evaluates both:
1. **Accuracy**: High mAP across all test data
2. **Consistency**: Low variation in performance across months (thermal drift robustness)

Example:
- Model A: mAP=0.60, CoV=0.10 → Score = 0.60 × 0.90 = **0.54**
- Model B: mAP=0.58, CoV=0.05 → Score = 0.58 × 0.95 = **0.55** ✅ Winner!

Model B wins despite lower mAP because it's more consistent across seasons.

### Per-Month Evaluation

Test set spans 5 months:
- June 2021
- July 2021
- August 2021
- September 2021
- October 2021

The competition computes:
1. mAP@0.5 for each month separately
2. Mean (μ) and standard deviation (σ) of monthly mAPs
3. CoV = σ / μ
4. Final score = overall_mAP × (1 - CoV)

## Generating Submissions

### Step 1: Train Model

Train your YOLOv8 model on LTDv2 training set:

```bash
python train_patched.py
```

After training completes, best weights saved to: `runs/detect/train/weights/best.pt`

### Step 2: Generate Predictions

Use the provided script:

```bash
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/test/images \
    --output submission_dev.json \
    --conf 0.001 \
    --device 0
```

Arguments:
- `--model`: Path to trained model (.pt file)
- `--test-dir`: Directory with test images
- `--output`: Output JSON filename
- `--conf`: Confidence threshold (use low value like 0.001, competition filters)
- `--device`: '0' for GPU, 'cpu' for CPU

For **development phase**, use validation images:
```bash
--test-dir data/ltdv2_full/valid/images
```

For **final phase**, use test images:
```bash
--test-dir data/ltdv2_full/test/images
```

### Step 3: Validate Submission

Before uploading, validate format:

```bash
python scripts/validate_submission.py submission_dev.json
```

This checks:
- ✓ Valid JSON format
- ✓ Required fields present
- ✓ Category IDs in range [1-4]
- ✓ Bounding box format correct
- ✓ Confidence scores in [0, 1]
- ✓ Image IDs match expected test set

**Output example:**
```
======================================================================
WACV 2026 RWS Submission Validator
======================================================================
File: submission_dev.json
======================================================================

📂 Loading submission file...
✅ Valid JSON format

🔍 Checking structure...
✅ List format: 125000 predictions

🔍 Validating predictions...

📊 Statistics:
   Total predictions: 125000
   Unique images: 41226
   Predictions per image: 3.0

   Per-class predictions:
     Person: 85000
     Bicycle: 5000
     Motorcycle: 10000
     Vehicle: 25000

   Score statistics:
     Min: 0.0010
     Max: 0.9950
     Mean: 0.4523

======================================================================
✅ VALIDATION PASSED: No errors found
======================================================================
```

### Step 4: Upload to Codabench

1. Go to: https://www.codabench.org/competitions/10954/
2. Click "Participate" tab
3. Select phase: "Development" or "Final Testing"
4. Upload `submission_dev.json`
5. Wait for evaluation (may take 10-30 minutes)
6. Check leaderboard for results

## Validation Checklist

Before submitting, ensure:

### Format
- [ ] File is valid JSON
- [ ] Root structure is a list (array), not object
- [ ] File size < 100MB (compress if needed: `gzip submission.json`)

### Required Fields
- [ ] Every prediction has `image_id`
- [ ] Every prediction has `category_id`
- [ ] Every prediction has `bbox`
- [ ] Every prediction has `score`

### Data Types
- [ ] `image_id` is integer
- [ ] `category_id` is integer in [1, 2, 3, 4]
- [ ] `bbox` is list of 4 floats
- [ ] `score` is float in [0.0, 1.0]

### Bounding Boxes
- [ ] Format is `[x, y, width, height]`, NOT `[x1, y1, x2, y2]`
- [ ] All coordinates are absolute pixels (not normalized)
- [ ] All values are non-negative
- [ ] Width and height are positive (> 0)

### Coverage
- [ ] All test images have at least one prediction (or empty if no detections)
- [ ] No predictions for images outside test set
- [ ] Image IDs match test set filenames

### Classes
- [ ] Only using 4 classes: Person(1), Bicycle(2), Motorcycle(3), Vehicle(4)
- [ ] No class ID 0 or class IDs > 4
- [ ] If using COCO pretrained: car/bus/truck mapped to Vehicle(4)

## Common Issues & Fixes

### Issue 1: Wrong Bbox Format

**Error**: Boxes appear in wrong location or fail validation

**Cause**: Using `[x1, y1, x2, y2]` instead of `[x, y, w, h]`

**Fix**:
```python
# Convert from xyxy to xywh
x1, y1, x2, y2 = box.xyxy[0]
x = float(x1)
y = float(y1)
width = float(x2 - x1)
height = float(y2 - y1)
bbox = [x, y, width, height]
```

### Issue 2: Missing Test Images

**Error**: Some test images have no predictions

**Cause**: Not generating predictions for images with no detections

**Fix**: Include all test images, even if predictions list is empty:
```python
# Always process every test image
for img_file in test_images:
    results = model(img_file)
    # Even if len(results.boxes) == 0, we've covered this image
```

### Issue 3: Wrong Class IDs

**Error**: Invalid category_id errors

**Cause**: Using YOLO class IDs (0-79) directly

**Fix**: Map YOLO classes to LTDv2 classes:
```python
YOLO_TO_LTD = {
    0: 1,  # person → Person
    1: 2,  # bicycle → Bicycle
    3: 3,  # motorcycle → Motorcycle
    2: 4,  # car → Vehicle
    5: 4,  # bus → Vehicle
    7: 4,  # truck → Vehicle
}
```

### Issue 4: Large File Size

**Error**: Upload fails due to file size

**Fix**: Compress with gzip:
```bash
gzip submission.json
# Upload submission.json.gz (Codabench accepts .gz files)
```

Or reduce predictions:
```python
# Filter low-confidence predictions
predictions = [p for p in predictions if p['score'] >= 0.01]
```

### Issue 5: Normalized Coordinates

**Error**: Tiny boxes all in corner

**Cause**: Using normalized coords [0, 1] instead of absolute pixels

**Fix**: Don't normalize! Use pixel values directly:
```python
# Correct: absolute pixels
bbox = [100.5, 200.3, 50.2, 80.1]

# Wrong: normalized
bbox = [0.15, 0.31, 0.08, 0.12]
```

## Development Phase Strategy

### 1. Quick Baseline Submission

Submit baseline model ASAP to:
- Test submission pipeline
- Get initial leaderboard position
- Identify potential issues early

### 2. Iterative Improvement

After baseline:
1. Analyze per-month performance (if provided)
2. Identify weak months
3. Add targeted data augmentation
4. Retrain and resubmit
5. Compare scores

### 3. Ensemble Strategy

For final submission:
- Train multiple models (different seeds, architectures)
- Ensemble predictions (average scores, NMS)
- Submit best ensemble

### 4. Test-Time Augmentation

Improve robustness:
```python
# Multi-scale inference
predictions_640 = model(img, imgsz=640)
predictions_800 = model(img, imgsz=800)
# Merge and NMS
```

## Final Phase Strategy

### 1. Model Selection

Choose best model from development phase:
- Highest development score
- Most consistent across months
- Best ensemble

### 2. Limited Submissions

Final phase has limited submissions!
- Use development phase to finalize model
- Only submit when confident
- Don't waste submissions on experiments

### 3. Confidence Threshold Tuning

Tune confidence threshold on validation set:
```python
for conf_thresh in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
    predictions = generate_predictions(model, val_set, conf=conf_thresh)
    score = evaluate(predictions)
    print(f"Conf {conf_thresh}: Score {score}")
```

Use best threshold for test set.

### 4. Final Checks

Before final submission:
- [ ] Validated with `validate_submission.py`
- [ ] Tested on development set first
- [ ] Confidence threshold optimized
- [ ] All test images covered
- [ ] File size acceptable
- [ ] Backed up model weights

## Submission Scripts Reference

### Generate Submission

```bash
# Development phase (validation set)
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/valid/images \
    --output submission_dev.json \
    --conf 0.001 \
    --device 0

# Final phase (test set)
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/test/images \
    --output submission_final.json \
    --conf 0.001 \
    --device 0
```

### Validate Submission

```bash
# Basic validation
python scripts/validate_submission.py submission_dev.json

# With test image ID verification
python scripts/validate_submission.py submission_dev.json --test-ids test_ids.txt
```

### Check File Size

```bash
# Check size
ls -lh submission_dev.json

# Compress if needed
gzip submission_dev.json
ls -lh submission_dev.json.gz
```

## Resources

### Competition
- Competition page: https://www.codabench.org/competitions/10954/
- Workshop website: https://vap.aau.dk/rws/
- Contact: asjo@create.aau.dk

### Dataset
- Download: HuggingFace (vapaau/LTDv2)
- Paper: TechRxiv 175339329.95323969
- Statistics: 1M+ frames, 8 months, 6.8M+ boxes

### Documentation
- Training approach: `docs/COMPETITION_APPROACH.md`
- Project README: `README.md`
- Quick reference: `docs/QUICK_REFERENCE.md`

## Quick Command Reference

```bash
# 1. Train model
python train_patched.py

# 2. Generate submission (development)
python scripts/generate_submission.py \
    --model runs/detect/train/weights/best.pt \
    --test-dir data/ltdv2_full/valid/images \
    --output submission_dev.json

# 3. Validate submission
python scripts/validate_submission.py submission_dev.json

# 4. Upload to Codabench
# Go to: https://www.codabench.org/competitions/10954/
# Click "Participate" → "Submit" → Upload JSON

# 5. Check leaderboard
# Go to: https://www.codabench.org/competitions/10954/
# Click "Results" tab
```

---

**Good luck with your submission! 🚀**
