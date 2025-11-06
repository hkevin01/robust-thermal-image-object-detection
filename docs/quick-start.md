# Quick Start Guide

Get up and running with the Robust Thermal Image Object Detection project in minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 50GB+ disk space for dataset

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/robust-thermal-image-object-detection.git
cd robust-thermal-image-object-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Download Dataset

### Option 1: Full Download

Download the complete LTDv2 dataset (~40GB):

```bash
python scripts/data/download_dataset.py --mode full --output_dir data/ltdv2
```

### Option 2: Streaming (Recommended for Development)

Use streaming mode to avoid downloading everything upfront:

```bash
python scripts/data/download_dataset.py --mode streaming --output_dir data/ltdv2 --max_samples 1000
```

## Setup Data Configuration

Create a `data.yaml` file for YOLOv8:

```bash
cp configs/data.yaml data/ltdv2/data.yaml
```

Edit `data/ltdv2/data.yaml` to match your directory structure:

```yaml
path: /absolute/path/to/data/ltdv2
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: Person
  1: Bicycle
  2: Motorcycle
  3: Vehicle
```

## Run Quick Tests

### Smoke Test

Verify everything is working:

```bash
python tests/smoke_test.py
```

Expected output:
```
============================================================
SMOKE TEST: Robust Thermal Image Object Detection
============================================================
Creating dummy thermal images...
Creating dummy annotations...

[1/4] Testing dataset loading...
✓ Dataset loaded successfully: 3 samples
✓ Sample format correct: torch.Size([3, 480, 640])

[2/4] Testing COCO format loading...
✓ COCO dataset loaded successfully: 3 samples

[3/4] Testing YOLOv8 model initialization...
✓ Model initialized successfully: yolov8n.pt

[4/4] Testing inference...
✓ Inference completed: 3 predictions
  - Image 0: 2 detections
  - Image 1: 1 detections
  - Image 2: 3 detections

============================================================
✓ ALL SMOKE TESTS PASSED
============================================================
```

### Unit Tests

```bash
pytest tests/unit/ -v
```

## Train Your First Model

### Baseline Training

Train a baseline YOLOv8 model:

```bash
python src/training/train.py \
  --config configs/baseline.yaml \
  --wandb-project thermal-detection
```

### Without W&B

If you don't want to use Weights & Biases:

```bash
python src/training/train.py \
  --config configs/baseline.yaml \
  --no-wandb
```

### Training Options

```bash
# Evaluation only (no training)
python src/training/train.py --config configs/baseline.yaml --eval-only

# Resume from checkpoint
python src/training/train.py --config configs/baseline.yaml --checkpoint runs/train/exp/weights/last.pt

# Use different config
python src/training/train.py --config configs/weather_conditioned.yaml
python src/training/train.py --config configs/domain_adaptation.yaml
```

## Monitor Training

### Weights & Biases

View training progress at: https://wandb.ai/your-username/thermal-detection

### Local Logs

Training outputs are saved to:
```
runs/
├── train/
│   └── exp/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.csv
│       └── args.yaml
```

## Run Inference

### On Images

```bash
from src.models.yolo_detector import ThermalYOLOv8

# Load model
model = ThermalYOLOv8.load("runs/train/exp/weights/best.pt")

# Run inference
results = model.predict(
    source="path/to/images",
    conf=0.25,
    iou=0.45,
    save=True
)
```

### On Video

```bash
results = model.predict(
    source="path/to/video.mp4",
    conf=0.25,
    iou=0.45,
    save=True
)
```

## Evaluate Model

```bash
from src.evaluation.metrics import TemporalDetectionMetrics

# Initialize metrics
metrics = TemporalDetectionMetrics(num_classes=4)

# Run validation
results = model.validate(data="data/ltdv2/data.yaml")

# Compute challenge score
challenge_score = metrics.compute_challenge_score()
print(f"Challenge Score: {challenge_score:.4f}")
```

## Hyperparameter Tuning

### W&B Sweep

```bash
# Initialize sweep
wandb sweep configs/wandb_sweep.yaml

# Run sweep agent
wandb agent your-entity/thermal-detection/sweep-id
```

### Manual Grid Search

Edit `configs/baseline.yaml` and run multiple experiments:

```bash
for lr in 0.0001 0.0005 0.001; do
  python src/training/train.py \
    --config configs/baseline.yaml \
    --wandb-project thermal-detection
done
```

## Docker Usage

### Build Container

```bash
docker build -t thermal-detection -f docker/Dockerfile .
```

### Run Training

```bash
docker run --gpus all -v $(pwd):/workspace thermal-detection \
  python src/training/train.py --config configs/baseline.yaml
```

## Common Issues

### Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Slow Training

- Enable mixed precision training
- Use multiple GPUs
- Reduce image size

### Import Errors

Add project root to Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Next Steps

1. **Explore Configs**: Check out different training strategies in `configs/`
2. **Read Documentation**: See `docs/project-plan.md` for detailed roadmap
3. **Join Community**: Participate in WACV 2026 RWS Challenge discussions
4. **Experiment**: Try different models, augmentations, and training techniques

## Useful Commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f runs/train/exp/train.log

# Clean up old runs
rm -rf runs/train/exp*
```

## Resources

- **Challenge Page**: [WACV 2026 RWS Challenge](https://www.kaggle.com/competitions/wacv-2026-rws-challenge)
- **Dataset**: [LTDv2 on HuggingFace](https://huggingface.co/datasets/ltdv2)
- **YOLOv8 Docs**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **W&B Guide**: [Weights & Biases Tutorial](https://docs.wandb.ai/)

## Support

For issues or questions:
1. Check `docs/project-plan.md` for detailed information
2. Review `tests/README.md` for testing help
3. Open an issue on GitHub
4. Contact the challenge organizers

Happy training! 🚀
