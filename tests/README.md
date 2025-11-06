# Tests

This directory contains all tests for the Robust Thermal Image Object Detection project.

## Test Structure

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_dataset.py    # Dataset loading and parsing tests
│   └── test_metrics.py    # Evaluation metrics tests
├── integration/        # Integration tests (TODO)
│   └── test_training.py   # Training pipeline tests
└── smoke_test.py      # Quick end-to-end smoke test
```

## Running Tests

### All Tests

Run all tests with pytest:

```bash
pytest tests/ -v
```

### With Coverage

Generate coverage report:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

View coverage in browser:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Specific Test Files

Run specific test file:

```bash
pytest tests/unit/test_dataset.py -v
pytest tests/unit/test_metrics.py -v
```

### Specific Test Cases

Run specific test case:

```bash
pytest tests/unit/test_dataset.py::TestLTDv2Dataset::test_init_csv -v
pytest tests/unit/test_metrics.py::TestTemporalDetectionMetrics::test_compute_challenge_score -v
```

### Smoke Test

Quick validation that everything works:

```bash
python tests/smoke_test.py
```

The smoke test:
1. Creates a minimal dummy dataset
2. Tests dataset loading (CSV and COCO formats)
3. Initializes YOLOv8 model
4. Runs inference on dummy images

This is useful for:
- Quick sanity checks during development
- CI/CD smoke testing
- Verifying environment setup

## Test Categories

### Unit Tests

#### `test_dataset.py` - Dataset Tests

Tests the `LTDv2Dataset` class:

- **Initialization**:
  - `test_init_csv`: CSV annotation loading
  - `test_init_coco_json`: COCO JSON annotation loading
  - `test_init_with_metadata`: Metadata file integration
  
- **Data Loading**:
  - `test_getitem`: Sample format validation
  - `test_getitem_with_metadata`: Metadata in samples
  - `test_bbox_normalization`: Bounding box format
  - `test_label_range`: Label validation
  
- **Error Handling**:
  - `test_missing_image_dir`: Missing directory handling
  - `test_missing_annotation_file`: Missing annotation handling
  - `test_corrupted_image_handling`: Retry logic for corrupted images
  - `test_empty_annotations`: Empty file handling
  
- **Statistics**:
  - `test_statistics`: Load time and failure tracking

**Total**: 15+ test cases

#### `test_metrics.py` - Metrics Tests

Tests the `TemporalDetectionMetrics` class:

- **Initialization**:
  - `test_init`: Metric configuration
  
- **Basic Operations**:
  - `test_update`: Adding predictions/targets
  - `test_reset`: Clearing internal state
  
- **Metric Computation**:
  - `test_compute_global_metrics`: Overall mAP calculation
  - `test_compute_monthly_metrics`: Per-month mAP
  - `test_compute_temporal_consistency`: CoV calculation
  - `test_compute_challenge_score`: Final challenge score
  - `test_per_class_metrics`: Per-class AP
  
- **Edge Cases**:
  - `test_empty_predictions`: No predictions handling
  - `test_empty_targets`: No targets handling
  - `test_no_temporal_ids`: Metrics without temporal info
  - `test_cov_with_single_month`: Single month CoV
  - `test_challenge_score_zero_map`: Zero mAP handling
  
- **Configuration**:
  - `test_different_iou_thresholds`: Multiple IoU values

**Total**: 18+ test cases

### Integration Tests (TODO)

#### `test_training.py` - Training Pipeline Tests

Planned tests:
- End-to-end training loop
- Checkpoint saving/loading
- W&B integration
- Validation during training
- Early stopping
- Learning rate scheduling

### Smoke Test

`smoke_test.py` - Quick End-to-End Test

A lightweight test that validates basic functionality:

1. **Dataset Creation**: Generates 3 dummy thermal images
2. **Dataset Loading**: Tests CSV and COCO format loading
3. **Model Initialization**: Creates YOLOv8n model
4. **Inference**: Runs predictions on dummy data

**Runtime**: ~10-30 seconds (depending on YOLOv8 download)

## Writing Tests

### Test Structure

```python
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import LTDv2Dataset

@pytest.fixture
def sample_data():
    """Create sample test data."""
    # Setup code
    yield data
    # Teardown code

class TestMyComponent:
    """Test suite for MyComponent."""
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Arrange
        component = MyComponent()
        
        # Act
        result = component.process(sample_data)
        
        # Assert
        assert result is not None
```

### Best Practices

1. **Use fixtures** for shared test data
2. **Name tests descriptively**: `test_<what>_<when>_<expected>`
3. **Test edge cases**: empty inputs, invalid data, boundary conditions
4. **Keep tests isolated**: Each test should be independent
5. **Use parametrize** for testing multiple inputs:

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply(input, expected):
    assert multiply(input) == expected
```

## CI/CD Integration

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests

See `.github/workflows/ci.yml` for CI configuration.

### CI Jobs

1. **Lint**: Code style checks (black, flake8, mypy, isort)
2. **Test**: Unit tests with coverage (Python 3.10, 3.11)
3. **Smoke Test**: Quick end-to-end validation

## Coverage Goals

Target coverage: **≥ 80%**

Current coverage:
- `src/data/dataset.py`: ~90%
- `src/evaluation/metrics.py`: ~85%
- `src/models/yolo_detector.py`: TODO
- `src/training/trainer.py`: TODO

## Troubleshooting

### Import Errors

If tests fail with import errors:

```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Missing Dependencies

Install test dependencies:

```bash
pip install pytest pytest-cov
```

### Slow Tests

Run with parallelization:

```bash
pip install pytest-xdist
pytest tests/ -n auto
```

### Debugging Tests

Run with verbose output and stop on first failure:

```bash
pytest tests/ -vv -x --pdb
```

## Contributing

When adding new code:

1. Write tests for new functionality
2. Ensure all tests pass locally
3. Check coverage doesn't decrease
4. Update this README if adding new test categories

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
