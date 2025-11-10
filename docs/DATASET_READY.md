# LTDv2 Dataset Ready for Training! 🎉

**Status**: ✅ COMPLETE
**Date**: November 6, 2025
**Total Time**: ~11 hours (download + extraction + conversion)

---

## 📊 Dataset Summary

### Final Statistics
- **Total Images**: 420,033
- **Training Set**: 329,299 images with 329,299 YOLO labels
- **Validation Set**: 43,850 images with 43,850 YOLO labels  
- **Test Set**: 46,884 images (no labels - for competition submission)
- **Classes**: 5 (background, person, bicycle, motorcycle, vehicle)
- **Total Annotations**: 2,912,990 bounding boxes
- **Format**: YOLO (normalized coordinates)

### Dataset Split Distribution
```
Train:      78.4% (329,299 images)
Validation: 10.4% (43,850 images)
Test:       11.2% (46,884 images - unlabeled)
```

### Storage
- **Location**: `data/ltdv2_full/`
- **Images**: Symlinks to extracted frames (no duplication)
- **Labels**: YOLO format .txt files
- **Total Size**: ~93 GB (frames + cache)

---

## 🔧 Technical Details

### Conversion Process
1. **Download**: 48 GB frames.zip + JSON annotations (13 minutes @ 2 GB/min)
2. **Extraction**: 1,069,247 raw thermal images (5 minutes)
3. **Conversion**: COCO JSON → YOLO format with unique filenames (3.5 minutes)
   - Flat filename structure: `YYYYMMDD_clip_X_TTTT_image_NNNN.jpg`
   - Preserves uniqueness by including date + clip + timestamp
   - Creates symlinks (instant) instead of copying files

### File Structure
```
data/ltdv2_full/
├── .cache/                    # HuggingFace cache (blobs)
├── frames/                    # Extracted thermal images (1M+)
│   └── frames/
│       ├── 20200514/
│       ├── 20200515/
│       └── ...
├── images/
│   ├── train/                 # 329,299 symlinks
│   ├── val/                   # 43,850 symlinks
│   └── test/                  # 46,884 symlinks
├── labels/
│   ├── train/                 # 329,299 YOLO .txt files
│   ├── val/                   # 43,850 YOLO .txt files
│   └── test/                  # (empty - no labels)
└── data.yaml                  # YOLO dataset config

```

### Sample Data
**Image**: `20200514_clip_22_2307_image_0015.jpg`
**Label**: `20200514_clip_22_2307_image_0015.txt`
```
1 0.212240 0.395833 0.023438 0.034722  # person
4 0.424479 0.345486 0.057292 0.059028  # vehicle
```

Format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

---

## ⚙️ System Configuration

### Current Setup
- **CPU**: AMD Ryzen 5 3600 (12 cores)
- **RAM**: 31 GB (6.7 GB used, 2.0 GB free)
- **GPU**: ❌ No NVIDIA GPU detected
- **PyTorch**: 2.9.0+cu128 (CUDA not available)
- **Ultralytics**: 8.3.225

### Performance Implications
⚠️ **Training on CPU will be VERY slow** (100-200x slower than GPU):
- Expected time for 1 epoch: ~8-12 hours on CPU vs ~3-5 minutes on GPU
- Full 100 epoch training: ~33-50 days on CPU vs ~5-8 hours on GPU

**Recommendation**: Use GPU for training, or significantly reduce dataset/epochs for CPU

---

## 🎯 Next Steps

### Immediate Actions

#### Option 1: GPU Training (Recommended)
If you have access to a GPU machine or cloud GPU:
```bash
# Transfer project to GPU machine
rsync -avz --exclude='.cache' --exclude='frames' data/ltdv2_full/ gpu-machine:/path/to/project/data/ltdv2_full/

# Or use cloud GPU (AWS, GCP, Colab Pro, etc.)
```

#### Option 2: CPU Training with Reduced Dataset
For testing/validation on CPU:
```bash
# Create small subset (10K train, 2K val)
./venv/bin/python scripts/data/download_ltdv2.py \
  --output data/ltdv2_subset \
  --mode subset \
  --subset-train 10000 \
  --subset-val 2000

# Train on subset
./venv/bin/python src/training/train.py \
  --config configs/baseline.yaml \
  --data data/ltdv2_subset/data.yaml \
  --epochs 10
```

#### Option 3: Start Full Training on CPU (Not Recommended)
Only if you have days/weeks to wait:
```bash
# Launch full training
./venv/bin/python src/training/train.py \
  --config configs/baseline.yaml \
  --wandb-project thermal-detection-ltdv2

# Monitor in another terminal
tail -f runs/train/baseline_yolov8m/train.log
```

### Training Configuration

**Baseline Config** (`configs/baseline.yaml`):
- Model: YOLOv8m (25M parameters)
- Epochs: 100
- Batch size: 16
- Image size: 640x640
- Optimizer: AdamW
- Learning rate: 0.01 → 0.0001

**Expected Results** (on GPU):
- Training time: ~5-8 hours
- mAP@0.5: 0.55-0.65
- CoV (temporal consistency): 0.20-0.30
- Challenge Score: 0.40-0.50

---

## 📝 Validation Checklist

- [x] Download LTDv2 dataset (48 GB)
- [x] Extract all frames (1M+ images)
- [x] Convert COCO → YOLO format
- [x] Verify image counts (420,033 total)
- [x] Verify label counts (373,149 with annotations)
- [x] Check label format (YOLO normalized)
- [x] Update baseline config
- [x] Test data loading
- [ ] **CRITICAL: Obtain GPU access for reasonable training time**
- [ ] Launch baseline training
- [ ] Monitor training metrics
- [ ] Evaluate temporal consistency
- [ ] Generate competition submission

---

## 🚀 Quick Start Commands

### Verify Dataset
```bash
# Check dataset integrity
ls -lh data/ltdv2_full/
cat data/ltdv2_full/data.yaml
find data/ltdv2_full/images -type l | wc -l

# Sample a few images/labels
ls data/ltdv2_full/images/train/ | head -5
cat data/ltdv2_full/labels/train/20200514_clip_22_2307_image_0015.txt
```

### Test Data Loading
```bash
# Quick test with 1 image
./venv/bin/python -c "
from src.data.ltd_dataset import LTDv2Dataset
import yaml

with open('data/ltdv2_full/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset = LTDv2Dataset(
    images_dir='data/ltdv2_full/images/train',
    labels_dir='data/ltdv2_full/labels/train',
    transform=None
)

print(f'Dataset size: {len(dataset)}')
img, target = dataset[0]
print(f'Image shape: {img.shape}')
print(f'Targets: {target}')
"
```

### Start Training (when GPU available)
```bash
# Baseline training
./venv/bin/python src/training/train.py \
  --config configs/baseline.yaml \
  --wandb-project thermal-detection-wacv2026

# With custom settings
./venv/bin/python src/training/train.py \
  --config configs/baseline.yaml \
  --epochs 50 \
  --batch-size 32 \
  --device 0
```

---

## 📈 Monitoring Training

### Real-time Logs
```bash
# Training progress
tail -f runs/train/baseline_yolov8m/train.log

# Watch GPU usage (if available)
watch -n 1 nvidia-smi
```

### Experiment Tracking
- **Local**: Results in `runs/train/baseline_yolov8m/`
- **W&B**: https://wandb.ai (if configured)
- **Metrics**: mAP, Precision, Recall, CoV, Challenge Score

---

## ⚠️ Known Issues & Solutions

### Issue: No GPU Available
**Impact**: Training 100-200x slower
**Solutions**:
1. Use cloud GPU (AWS g4dn, GCP T4, Colab Pro)
2. Reduce dataset to 10K images for CPU testing
3. Use smaller model (YOLOv8n) and fewer epochs (10-20)

### Issue: Out of Memory
**Solutions**:
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Reduce image size: `--img-size 512` or `--img-size 416`
- Use smaller model: `yolov8n` or `yolov8s`

### Issue: Training Too Slow
**Solutions**:
- Enable mixed precision: `--amp` (automatic in YOLO)
- Increase batch size (on GPU): `--batch-size 32` or `--batch-size 64`
- Use multi-GPU: `--device 0,1,2,3`

---

## 📚 Documentation

- **Project Docs**: `docs/`
- **Config Examples**: `configs/`
- **Training Scripts**: `src/training/`
- **Data Scripts**: `scripts/data/`
- **Experiment Log**: `docs/experiment-log.md`

---

## 🎓 Competition Details

**Challenge**: WACV 2026 RWS - Robust Thermal Object Detection
**Metric**: mAP@0.5 × (1 - CoV)
- mAP@0.5: Standard object detection accuracy
- CoV: Coefficient of Variation (temporal consistency penalty)
- Lower CoV = more consistent predictions over time = higher score

**Submission**:
- Format: COCO JSON predictions on test set
- Test set: 46,884 unlabeled images
- Deadline: TBD

---

## ✅ Conclusion

The LTDv2 dataset is **fully prepared and ready for training**! 

**Current Status**: ✅ COMPLETE
- ✅ 420,033 images converted
- ✅ YOLO format labels created
- ✅ Dataset configuration updated
- ✅ System verified

**Blocker**: ⚠️ No GPU available - training will be extremely slow on CPU

**Recommended Next Action**: 
1. **Obtain GPU access** (cloud or local)
2. Transfer dataset to GPU machine
3. Launch baseline training
4. Monitor and iterate

Good luck with the competition! 🚀
