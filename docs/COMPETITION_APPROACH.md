# WACV 2026 RWS Challenge: Competition Approach & Architecture

## Competition Details

**Challenge**: [WACV 2026 Real World Surveillance Workshop](https://competitions.codalab.org/competitions/36713)

**Goal**: Build thermal object detectors that maintain consistent performance across:
- **Temporal variations**: 8 months of data (March-October 2021)
- **Seasonal changes**: Spring, Summer, Fall
- **Weather conditions**: Clear, Rainy, Foggy
- **Time of day**: Day and Night recordings

**Evaluation Metric**:
```
Robustness Score = mAP@0.5 × (1 - CoV)
```

Where:
- `mAP@0.5`: Detection accuracy at 0.5 IoU threshold
- `CoV`: Coefficient of Variation (std_dev / mean) across temporal bins
- Lower CoV = More consistent across conditions = Higher Robustness Score

---

## YOLOv8 Architecture: Deep Dive

### 1. Backbone: CSPDarknet with Cross-Stage Partial Connections

**Purpose**: Extract hierarchical features from thermal images

**How CSP Works**:
```
Input Feature Map
    ↓
Split into two branches:
    ├── Branch A: Goes through Conv + Residual blocks
    └── Branch B: Bypasses directly (partial connection)
    ↓
Concatenate branches
    ↓
Output Feature Map
```

**Benefits for Thermal Robustness**:
- **Reduced redundancy**: Partial connections prevent recomputing similar features
- **Gradient flow**: Direct bypass ensures strong gradients during backprop
- **Multi-scale learning**: Different branches capture different abstraction levels
- **Thermal drift handling**: Learns both low-level thermal patterns AND high-level semantic features

**Architecture Stages**:
```
Input (640×640×3)
    ↓
Stem: Conv 6×6, stride=2 → 320×320×32
    ↓
Stage 1: CSP Block → 320×320×64 (C2)
    ↓
Stage 2: CSP Block → 160×160×128 (C3) ← Fed to PANet
    ↓
Stage 3: CSP Block → 80×80×256 (C4) ← Fed to PANet
    ↓
Stage 4: CSP Block + SPPF → 40×40×512 (C5) ← Fed to PANet
```

**SPPF (Spatial Pyramid Pooling - Fast)**:
- Pools features at multiple scales (5×5, 9×9, 13×13)
- Captures both local details and global context
- Critical for handling objects at varying thermal contrasts

---

### 2. Neck: PANet (Path Aggregation Network)

**Purpose**: Fuse multi-scale features for robust detection at all scales

**Architecture Flow**:
```
Backbone Features:
    C3 (160×160×128)
    C4 (80×80×256)
    C5 (40×40×512)
        ↓
┌───────────────────────────────────┐
│  TOP-DOWN PATHWAY (FPN-style)     │
│  C5 → Upsample → Fuse with C4     │
│  C4' → Upsample → Fuse with C3    │
│  Adds semantic info to low levels │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  BOTTOM-UP PATHWAY (Aggregation)  │
│  C3' → Downsample → Fuse with C4' │
│  C4'' → Downsample → Fuse with C5'│
│  Adds localization to high levels │
└───────────────────────────────────┘
        ↓
Three Feature Pyramids:
    P3 (80×80) - Small objects
    P4 (40×40) - Medium objects
    P5 (20×20) - Large objects
```

**Why This Matters for Thermal**:
- **Top-down**: High-level features help classify "what" despite thermal signature changes
- **Bottom-up**: Low-level features maintain precise "where" localization
- **Multi-scale**: Handles varying thermal contrast at different object distances

**Detection Grid Scales**:
| Scale | Grid Size | Receptive Field | Object Types | Example Thermal Objects |
|-------|-----------|-----------------|--------------|-------------------------|
| P3 | 80×80 | Small | 8-40 px | Distant pedestrians, motorcycles |
| P4 | 40×40 | Medium | 32-128 px | Cars, close pedestrians |
| P5 | 20×20 | Large | 128+ px | Buses, trucks, very close vehicles |

---

### 3. Head: Decoupled Classification & Localization

**Traditional YOLO Heads** (Coupled):
```
Features → Conv → Conv → [class_scores, x, y, w, h]
                          ↑
                 Single head predicts both
```

**YOLOv8 Heads** (Decoupled):
```
Features ┬→ Classification Branch → Conv → Conv → [class_scores]
         │   (learns "WHAT")
         │
         └→ Localization Branch → Conv → Conv → [x, y, w, h]
             (learns "WHERE")
```

**Why Decoupling Helps**:
1. **Task Conflict Resolution**:
   - Classification needs: High-level semantic features (person vs car)
   - Localization needs: Low-level spatial features (exact edges, corners)
   - Sharing layers creates optimization conflicts

2. **Thermal Robustness Benefits**:
   - **Classification branch**: Learns thermal-invariant object identity
     - Example: "Person" signature changes with temperature, but shape/context patterns persist
   - **Localization branch**: Focuses purely on where thermal edges are
     - Independent of whether object is "hot" or "cold" relative to background

3. **Training Efficiency**:
   - Each branch can optimize independently
   - Faster convergence (empirically 20-30% faster training)
   - Better gradient flow to specific tasks

**Head Architecture Details**:
```python
# Classification Branch (per scale: P3, P4, P5)
Conv2d(256, 256, 3×3) + BatchNorm + SiLU
Conv2d(256, 256, 3×3) + BatchNorm + SiLU
Conv2d(256, num_classes, 1×1)  # Output: [B, 5, H, W]

# Localization Branch (per scale)
Conv2d(256, 256, 3×3) + BatchNorm + SiLU
Conv2d(256, 256, 3×3) + BatchNorm + SiLU
Conv2d(256, 64, 1×1)  # DFL representation
# Post-process to [B, 4, H, W] for (x, y, w, h)
```

---

### 4. Training: Task-Aligned Learning (TAL)

**Problem with Traditional Anchor-Based Methods**:
```
Pre-defined anchors: [small, medium, large] × 3 aspect ratios = 9 anchors/scale
Assignment: IoU >= 0.5 → Positive, else Negative

Issues:
❌ Fixed anchors don't adapt to thermal signature size variations
❌ High classification score + poor localization = False confidence
❌ Many suboptimal positive samples dilute training signal
```

**YOLOv8's Anchor-Free TAL Solution**:
```
For each ground truth box:
    1. Compute alignment metric for ALL predictions in nearby cells:
       
       alignment_score = (classification_score ^ α) × (IoU ^ β)
       
       Where α=1.0, β=6.0 (heavily weights localization quality)
    
    2. Select top-k predictions with highest alignment
       (typically k=10 per ground truth)
    
    3. Assign these as positive samples, rest as negative
    
    4. Use soft labels based on alignment score (not hard 0/1)
```

**TAL Benefits for Thermal Robustness**:

1. **Dynamic Assignment**:
   - No fixed anchor sizes to tune for each thermal condition
   - Adapts to varying object scales automatically
   - Example: Same car appears larger in cold background vs warm background

2. **Quality-Aware Training**:
   - Only predictions that are BOTH confident AND accurate get strong gradients
   - Prevents model from being overconfident on poorly localized detections
   - Critical when thermal signatures are ambiguous

3. **Soft Label Distribution**:
   - Instead of hard 0/1 labels, uses alignment score as weight
   - Predictions close to optimal get some gradient signal
   - More robust learning signal, especially early in training

**Mathematical Formulation**:
```
Given:
- Ground truth box: g = (x_g, y_g, w_g, h_g, class_g)
- Prediction: p = (x_p, y_p, w_p, h_p, class_p)
- Classification score: s_cls
- IoU: IoU(p, g)

Alignment metric:
A(p, g) = s_cls^α × IoU(p, g)^β

Positive sample selection:
positives = top_k(A, k=10)

Loss weight for positive sample i:
w_i = A_i / Σ(A_j for j in positives)
```

---

## Our Complete Training Pipeline

### Phase 1: Baseline Training (Current - Epoch 1/50)

**Configuration**:
```yaml
Model: YOLOv8n (3.2M parameters)
Dataset: LTDv2 full training set (329,299 images)
Batch Size: 4
Image Size: 640×640
Optimizer: SGD (momentum=0.937, weight_decay=0.0005)
Learning Rate: 0.01 initial, cosine decay
Epochs: 50
Workers: 8 (parallel data loading)
Mixed Precision: Disabled (for stability)
```

**Augmentations** (Active):
- Mosaic (4-image composition): 100% until epoch 40
- Horizontal flip: 50%
- Scale: 0.5× to 1.5×
- Translate: ±10%
- HSV jitter: Hue ±0.015, Saturation ±0.7, Value ±0.4
- Rotation: ±0° (disabled for now)

**Loss Function**:
```
L_total = 7.5 × L_box + 0.5 × L_cls + 1.5 × L_dfl

Where:
- L_box: CIoU loss (Complete IoU, includes overlap + distance + aspect ratio)
- L_cls: Binary Cross-Entropy with TAL-weighted samples
- L_dfl: Distribution Focal Loss for bbox refinement
```

**Current Progress**:
- ✅ Training stable at 5 batches/second
- ✅ Losses decreasing: Box ↓23%, Class ↓58%, DFL ↓26%
- ✅ No crashes, GPU running smoothly
- ⏳ ETA: 4 hours per epoch, ~9 days total

---

### Phase 2: Robustness Enhancements (Planned)

**1. Temporal Stratified Sampling**:
```python
# Group dataset by metadata
bins = {
    'winter_day': images from Dec-Feb, 6am-6pm,
    'winter_night': images from Dec-Feb, 6pm-6am,
    'spring_day': images from Mar-May, 6am-6pm,
    # ... etc for 12-16 bins
}

# Ensure each batch has diverse conditions
for each batch:
    sample_count = batch_size // len(bins)
    batch = []
    for bin in bins:
        batch += random_sample(bin, sample_count)
```

**2. Metadata-Augmented Training**:
```python
# Add temporal context as auxiliary input
def forward(image, metadata):
    # Metadata: [month, hour, temperature, weather_code]
    temporal_embed = embedding_layer(metadata)  # → 64-dim vector
    
    # Inject into backbone at multiple scales
    features_C3 = backbone.stage2(x + temporal_embed)
    # ... continue with standard forward pass
```

**3. Consistency-Aware Loss**:
```python
# Track per-condition performance during training
condition_losses = defaultdict(list)

for batch in dataloader:
    loss = standard_loss(predictions, targets)
    
    # Group by condition
    for idx, condition in enumerate(batch_conditions):
        condition_losses[condition].append(loss[idx])
    
    # Add consistency penalty
    condition_stds = [std(losses) for losses in condition_losses.values()]
    consistency_penalty = mean(condition_stds)
    
    total_loss = loss + λ × consistency_penalty
```

**4. Test-Time Augmentation**:
```python
# Inference with multiple augmentations
def robust_predict(image):
    predictions = []
    
    # Original
    predictions.append(model(image))
    
    # Horizontal flip
    predictions.append(model(flip(image)))
    
    # Multi-scale
    for scale in [0.83, 1.0, 1.17]:
        resized = resize(image, scale)
        predictions.append(model(resized))
    
    # Aggregate (weighted voting or NMS across all)
    return ensemble_predictions(predictions)
```

---

### Phase 3: Validation & Submission

**Robustness Score Computation**:
```python
# Split validation set into temporal bins
val_bins = split_by_temporal_metadata(val_set)

# Evaluate each bin
bin_mAPs = {}
for bin_name, bin_images in val_bins.items():
    predictions = model.predict(bin_images)
    bin_mAPs[bin_name] = compute_mAP(predictions, bin_images)

# Compute overall metrics
overall_mAP = mean(bin_mAPs.values())
CoV = std(bin_mAPs.values()) / mean(bin_mAPs.values())
Robustness_Score = overall_mAP * (1 - CoV)

print(f"mAP@0.5: {overall_mAP:.4f}")
print(f"CoV: {CoV:.4f}")
print(f"Robustness Score: {Robustness_Score:.4f}")
```

**Test Set Submission**:
```python
# Generate predictions on held-out test set
test_images = load_test_set()  # 46,884 images (June-Oct 2021)
predictions = []

for image in test_images:
    pred = model.predict(image)  # Or robust_predict() with TTA
    predictions.append({
        'image_id': image.id,
        'category_id': pred.class_id,
        'bbox': pred.bbox,  # [x, y, width, height]
        'score': pred.confidence
    })

# Save in COCO JSON format
save_coco_json(predictions, 'submission.json')

# Upload to CodaLab
# https://competitions.codalab.org/competitions/36713#participate-submit_results
```

---

## Key Decisions & Rationale

### Why YOLOv8n (not YOLOv8m/l/x)?

**Pros**:
- ✅ Fast training (5 batches/sec vs 2-3 for larger models)
- ✅ Less overfitting risk (fewer parameters)
- ✅ More training iterations in limited time
- ✅ Sufficient capacity for 5 classes

**Cons**:
- ❌ Lower peak accuracy (~2-3% mAP vs YOLOv8m)
- ❌ May miss very small objects

**Decision**: Start with YOLOv8n for rapid experimentation. If time permits and baseline is strong, try YOLOv8s/m.

---

### Why Batch Size 4?

**Pros**:
- ✅ Stable gradients (more samples = less noise)
- ✅ Better generalization (regularization effect)
- ✅ Fits in 6GB VRAM with augmentation overhead

**Cons**:
- ❌ Slower training (fewer batches processed)

**Decision**: Prioritize stability and generalization over raw speed. Thermal robustness requires smooth optimization landscape.

---

### Why Disable Mixed Precision?

**Pros of FP16**:
- ✅ 2× faster training
- ✅ Reduced memory usage

**Cons**:
- ❌ Potential numerical instability
- ❌ Loss scale tuning required
- ❌ May affect small gradient signals

**Decision**: Disable AMP for now to ensure stable training. Can enable later if training is stable and need speed boost.

---

## Competition Timeline

| Phase | Duration | Goal | Status |
|-------|----------|------|--------|
| **Baseline Training** | Days 1-10 | Get first mAP@0.5 result | 🔄 In Progress (Day 1) |
| **Hyperparameter Tuning** | Days 11-15 | Optimize batch size, LR, augmentation | ⏳ Pending |
| **Robustness Enhancements** | Days 16-25 | Add temporal features, consistency loss | ⏳ Pending |
| **Validation Analysis** | Days 26-28 | Compute CoV, analyze per-condition performance | ⏳ Pending |
| **Test Set Evaluation** | Days 29-30 | Generate submission, upload to CodaLab | ⏳ Pending |
| **Iteration & Refinement** | Ongoing | Improve based on leaderboard feedback | ⏳ Pending |

---

## References

- **YOLOv8 Paper**: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- **LTDv2 Dataset**: [arXiv:2108.08633](https://arxiv.org/abs/2108.08633)
- **WACV 2026 RWS**: [Competition Page](https://competitions.codalab.org/competitions/36713)
- **PANet Paper**: [arXiv:1803.01534](https://arxiv.org/abs/1803.01534)
- **CSPNet Paper**: [arXiv:1911.11929](https://arxiv.org/abs/1911.11929)
- **Task-Aligned Learning**: Part of YOLOv8 (not separate paper)
