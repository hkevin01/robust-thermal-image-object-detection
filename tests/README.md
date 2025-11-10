# Submission System Test Suite

Comprehensive testing for the WACV 2026 RWS competition submission system.

## Test Files

### Unit Tests
**File**: `test_submission_system.py`

Tests core submission system components:
- ✅ Submission format validation
- ✅ Class mapping (YOLO → LTDv2)
- ✅ Bounding box conversion (xyxy → xywh)
- ✅ Robustness score calculation
- ✅ Edge cases and boundary conditions

**Run**:
```bash
python tests/test_submission_system.py
```

### Integration Tests
**File**: `test_integration.py`

Tests actual scripts end-to-end:
- ✅ `generate_submission.py` script
- ✅ `validate_submission.py` script
- ✅ Data configuration (`data.yaml`)
- ✅ Documentation completeness

**Run**:
```bash
python tests/test_integration.py
```

### Pre-Submission Tests
**File**: `test_pre_submission.py`

Tests pre-submission validation checklist:
- ✅ Valid submissions pass
- ✅ Invalid structures detected
- ✅ Category ID validation
- ✅ Bbox format validation
- ✅ Score range validation
- ✅ Missing fields detection
- ✅ Statistics calculation

**Run**:
```bash
python tests/test_pre_submission.py
```

## Run All Tests

### Using Test Runner
```bash
./tests/run_all_tests.sh
```

### Manual
```bash
cd tests
python test_submission_system.py
python test_integration.py
python test_pre_submission.py
```

## Test Coverage

### What's Tested

#### ✅ Format Validation
- [x] JSON structure (list vs dict)
- [x] Required fields present
- [x] Field types correct
- [x] Field values in valid ranges
- [x] No duplicate IDs

#### ✅ Competition Requirements
- [x] 4 classes (Person, Bicycle, Motorcycle, Vehicle)
- [x] Category IDs 1-4 (not 0-3)
- [x] Bbox format [x, y, w, h] (not [x1, y1, x2, y2])
- [x] Absolute pixel coordinates (not normalized)
- [x] Confidence scores [0.0, 1.0]

#### ✅ Class Mapping
- [x] YOLO person (0) → LTDv2 Person (1)
- [x] YOLO bicycle (1) → LTDv2 Bicycle (2)
- [x] YOLO motorcycle (3) → LTDv2 Motorcycle (3)
- [x] YOLO car/bus/truck (2/5/7) → LTDv2 Vehicle (4)

#### ✅ Bbox Conversion
- [x] xyxy → xywh conversion
- [x] Positive width/height enforcement
- [x] Area preservation
- [x] Coordinate validation

#### ✅ Edge Cases
- [x] Empty submissions
- [x] Large submissions (200K+ predictions)
- [x] Tiny bounding boxes
- [x] Boundary score values (0.0, 1.0)
- [x] Float vs int handling

#### ✅ System Integration
- [x] Scripts exist and executable
- [x] Documentation complete
- [x] Data configuration correct
- [x] All required files present

## Expected Test Results

### Unit Tests
- **Tests**: 25
- **Expected Pass**: 25
- **Expected Fail**: 0

### Integration Tests
- **Tests**: 13
- **Expected Pass**: 13
- **Expected Fail**: 0

### Pre-Submission Tests
- **Tests**: 9
- **Expected Pass**: 9
- **Expected Fail**: 0

### Total
- **Total Tests**: 47
- **All Should Pass**: ✅

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install ultralytics torch numpy
      - name: Run tests
        run: |
          ./tests/run_all_tests.sh
```

## Troubleshooting

### Tests Fail

1. **Check Python version**: Requires Python 3.8+
   ```bash
   python --version
   ```

2. **Install dependencies**:
   ```bash
   pip install ultralytics torch numpy tqdm
   ```

3. **Check file permissions**:
   ```bash
   chmod +x tests/*.py tests/*.sh
   ```

4. **Run individual test**:
   ```bash
   python tests/test_submission_system.py -v
   ```

### Import Errors

Make sure you're running from project root:
```bash
cd /path/to/robust-thermal-image-object-detection
python tests/test_submission_system.py
```

### Module Not Found

Add project to PYTHONPATH:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python tests/test_integration.py
```

## Pre-Upload Checklist

Before every competition submission:

1. **Run all tests**:
   ```bash
   ./tests/run_all_tests.sh
   ```

2. **Run pre-submission validation**:
   ```bash
   python scripts/pre_submission_check.py submission_dev.json
   ```

3. **Check validation output**:
   - ✅ All checks pass
   - ⚠️ Review warnings
   - ❌ Fix any errors

4. **Upload to Codabench**:
   - https://www.codabench.org/competitions/10954/

## Test Maintenance

### Adding New Tests

1. Create test class in appropriate file
2. Follow naming convention: `test_<feature>_<condition>`
3. Use descriptive docstrings
4. Include both positive and negative cases
5. Update this README

### Updating Tests

When submission requirements change:
1. Update validation logic
2. Update test expectations
3. Update documentation
4. Run full test suite
5. Update README if needed

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./tests/run_all_tests.sh` | Run complete test suite |
| `python tests/test_submission_system.py` | Unit tests only |
| `python tests/test_integration.py` | Integration tests only |
| `python tests/test_pre_submission.py` | Pre-submission validation tests |
| `python scripts/pre_submission_check.py submission.json` | Validate submission file |
| `python scripts/validate_submission.py submission.json` | Quick validation |
| `python scripts/generate_submission.py --help` | Generation help |

## Resources

- **Competition**: https://www.codabench.org/competitions/10954/
- **Documentation**: `docs/COMPETITION_SUBMISSION_GUIDE.md`
- **Workflow**: `docs/SUBMISSION_WORKFLOW.md`
- **Checklist**: `docs/SUBMISSION_CHECKLIST.md`

---

**Last Updated**: Nov 10, 2025  
**Test Suite Version**: 1.0  
**Competition**: WACV 2026 RWS Thermal Object Detection
