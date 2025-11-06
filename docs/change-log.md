# Change Log

All notable changes to the Robust Thermal Image Object Detection project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Project Structure**: Complete project setup with memory-bank, src layout, tests, docs, scripts, configs
- **Dataset Loader** (`src/data/dataset.py`):
  - `LTDv2Dataset` class with robust error handling
  - CSV and JSON annotation format support
  - COCO JSON format parsing with `_load_coco_annotations()`
  - Image loading with exponential backoff retry logic
  - Metadata integration (temperature, humidity, solar radiation)
  - Boundary condition validation
  - Statistics tracking (load times, failures)
  
- **Evaluation Metrics** (`src/evaluation/metrics.py`):
  - `TemporalDetectionMetrics` class for challenge-specific evaluation
  - Global mAP computation with per-class breakdown
  - Monthly metrics tracking for temporal analysis
  - Coefficient of Variation (CoV) computation
  - Challenge score calculation: `mAP@0.5 × (1 - CoV)`
  
- **YOLOv8 Model Wrapper** (`src/models/yolo_detector.py`):
  - `ThermalYOLOv8` class wrapping Ultralytics YOLO
  - Training with 40+ hyperparameters
  - Inference with Test-Time Augmentation support
  - Model validation and metrics extraction
  - Save/load/export functionality
  - `create_data_yaml()` helper for YOLO data config
  
- **Training Pipeline** (`src/training/trainer.py`, `src/training/train.py`):
  - `ThermalDetectorTrainer` orchestrator class
  - Weights & Biases integration with graceful fallback
  - Complete training loop with configurable parameters
  - Evaluation mode for validation/test sets
  - YAML configuration loading/saving
  - Command-line interface with argparse
  
- **Configuration Files**:
  - `configs/baseline.yaml`: Baseline YOLOv8 training config
  - `configs/wandb_sweep.yaml`: Bayesian hyperparameter sweep config
  - `configs/data.yaml`: YOLO data configuration template
  - `configs/weather_conditioned.yaml`: Weather-aware training with metadata fusion
  - `configs/domain_adaptation.yaml`: Temporal consistency and domain adaptation
  
- **Unit Tests**:
  - `tests/unit/test_dataset.py`: Comprehensive dataset tests (15+ test cases)
    - CSV and COCO format loading
    - Metadata integration
    - Corrupted image handling
    - Bbox normalization and validation
    - Empty data handling
    - Statistics computation
  - `tests/unit/test_metrics.py`: Metrics evaluation tests (18+ test cases)
    - Global mAP computation
    - Monthly metrics tracking
    - Temporal consistency (CoV)
    - Challenge score calculation
    - Edge cases (empty predictions/targets)
    - Per-class metrics
    - Different IoU thresholds
  
- **Smoke Test** (`tests/smoke_test.py`):
  - End-to-end integration test
  - Creates dummy thermal dataset
  - Tests dataset loading (CSV and COCO)
  - Tests YOLOv8 initialization
  - Tests inference pipeline
  
- **CI/CD** (`.github/workflows/ci.yml`):
  - Lint job: black, flake8, mypy, isort
  - Test job: pytest with coverage on Python 3.10, 3.11
  - **Smoke test job**: Validates end-to-end functionality
  - Codecov integration
  
- **Dataset Download Script** (`scripts/data/download_dataset.py`):
  - Streaming download from HuggingFace
  - Full dataset download option
  - Selective file download
  - Progress tracking
  
- **Documentation**:
  - `README.md`: Complete project overview with setup instructions
  - `docs/project-plan.md`: 10-phase development roadmap
  - `memory-bank/app-description.md`: Challenge requirements and dataset info
  - VS Code settings with Copilot auto-approval
  - Docker setup with venv

### Changed
- N/A (initial release)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## Version History

### [0.1.0] - 2024-01-XX - Initial Release
- Complete project structure
- Baseline YOLOv8 implementation
- Comprehensive testing suite
- CI/CD pipeline
- Configuration files for multiple training strategies

---

## Notes

### Testing Coverage
- Unit tests: Dataset, Metrics
- Integration tests: Smoke test (end-to-end)
- CI/CD: Automated testing on push/PR

### Configuration Files
- `baseline.yaml`: Standard YOLOv8 training
- `weather_conditioned.yaml`: Metadata fusion approach
- `domain_adaptation.yaml`: Temporal consistency focus
- `wandb_sweep.yaml`: Hyperparameter optimization

### Future Additions
- Integration tests for training pipeline
- Performance benchmarks
- Model zoo with pretrained checkpoints
- Visualization tools
- Leaderboard submission scripts
