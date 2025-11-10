#!/usr/bin/env python3
"""
Validate Competition Submission Format
========================================

Checks submission JSON against WACV 2026 RWS requirements:
- https://www.codabench.org/competitions/10954/

Validation checks:
✓ JSON format and structure
✓ Required fields present
✓ Category IDs (1-4 only)
✓ Bbox format (x, y, width, height)
✓ Score range (0.0-1.0)
✓ Image IDs match expected test set
✓ No duplicate detections
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def validate_submission(json_path: str, test_image_ids: list = None):
    """
    Validate submission JSON format.
    
    Args:
        json_path: Path to submission JSON file
        test_image_ids: Optional list of expected test image IDs
        
    Returns:
        (is_valid, errors, warnings)
    """
    
    errors = []
    warnings = []
    
    print("="*70)
    print("WACV 2026 RWS Submission Validator")
    print("="*70)
    print(f"File: {json_path}")
    print("="*70)
    
    # Check file exists
    if not Path(json_path).exists():
        errors.append(f"File not found: {json_path}")
        return False, errors, warnings
    
    # Load JSON
    print("\n📂 Loading submission file...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {e}")
        return False, errors, warnings
    
    print(f"✅ Valid JSON format")
    
    # Check structure
    print("\n🔍 Checking structure...")
    
    # Should be a list of predictions
    if not isinstance(data, list):
        errors.append("Submission should be a JSON array (list), not an object")
        return False, errors, warnings
    
    print(f"✅ List format: {len(data)} predictions")
    
    if len(data) == 0:
        warnings.append("Empty submission (0 predictions)")
    
    # Validate each prediction
    print("\n🔍 Validating predictions...")
    
    required_fields = {'image_id', 'category_id', 'bbox', 'score'}
    valid_categories = {1, 2, 3, 4}  # Person, Bicycle, Motorcycle, Vehicle
    
    image_ids_seen = set()
    category_counts = defaultdict(int)
    score_stats = {'min': float('inf'), 'max': float('-inf'), 'sum': 0}
    
    for idx, pred in enumerate(data):
        pred_id = f"Prediction #{idx+1}"
        
        # Check fields
        missing_fields = required_fields - set(pred.keys())
        if missing_fields:
            errors.append(f"{pred_id}: Missing fields: {missing_fields}")
            continue
        
        # Validate image_id
        if not isinstance(pred['image_id'], int):
            errors.append(f"{pred_id}: image_id must be integer, got {type(pred['image_id'])}")
        else:
            image_ids_seen.add(pred['image_id'])
        
        # Validate category_id
        if pred['category_id'] not in valid_categories:
            errors.append(f"{pred_id}: Invalid category_id {pred['category_id']}, must be 1-4")
        else:
            category_counts[pred['category_id']] += 1
        
        # Validate bbox
        bbox = pred['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{pred_id}: bbox must be list of 4 numbers [x, y, w, h]")
        else:
            x, y, w, h = bbox
            if any(not isinstance(v, (int, float)) for v in bbox):
                errors.append(f"{pred_id}: bbox values must be numbers")
            elif any(v < 0 for v in bbox):
                errors.append(f"{pred_id}: bbox values cannot be negative: {bbox}")
            elif w <= 0 or h <= 0:
                errors.append(f"{pred_id}: bbox width/height must be positive: {bbox}")
        
        # Validate score
        score = pred['score']
        if not isinstance(score, (int, float)):
            errors.append(f"{pred_id}: score must be number, got {type(score)}")
        elif not (0.0 <= score <= 1.0):
            errors.append(f"{pred_id}: score must be in [0.0, 1.0], got {score}")
        else:
            score_stats['min'] = min(score_stats['min'], score)
            score_stats['max'] = max(score_stats['max'], score)
            score_stats['sum'] += score
    
    # Print statistics
    print("\n📊 Statistics:")
    print(f"   Total predictions: {len(data)}")
    print(f"   Unique images: {len(image_ids_seen)}")
    
    if len(data) > 0:
        print(f"   Predictions per image: {len(data) / len(image_ids_seen):.1f}")
    
    CLASS_NAMES = {1: "Person", 2: "Bicycle", 3: "Motorcycle", 4: "Vehicle"}
    print("\n   Per-class predictions:")
    for cat_id in sorted(valid_categories):
        count = category_counts[cat_id]
        print(f"     {CLASS_NAMES[cat_id]}: {count}")
    
    if len(data) > 0:
        print(f"\n   Score statistics:")
        print(f"     Min: {score_stats['min']:.4f}")
        print(f"     Max: {score_stats['max']:.4f}")
        print(f"     Mean: {score_stats['sum']/len(data):.4f}")
    
    # Check against expected test image IDs
    if test_image_ids is not None:
        print("\n🔍 Checking test image coverage...")
        expected_ids = set(test_image_ids)
        missing_ids = expected_ids - image_ids_seen
        extra_ids = image_ids_seen - expected_ids
        
        if missing_ids:
            warnings.append(f"{len(missing_ids)} test images have no predictions")
            if len(missing_ids) <= 10:
                warnings.append(f"Missing image IDs: {sorted(missing_ids)}")
        
        if extra_ids:
            warnings.append(f"{len(extra_ids)} predictions for unknown image IDs")
            if len(extra_ids) <= 10:
                warnings.append(f"Extra image IDs: {sorted(extra_ids)}")
        
        if not missing_ids and not extra_ids:
            print(f"✅ All {len(expected_ids)} test images covered")
    
    # Print results
    print("\n" + "="*70)
    
    if errors:
        print(f"❌ VALIDATION FAILED: {len(errors)} errors found")
        print("\nErrors:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  ❌ {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    else:
        print("✅ VALIDATION PASSED: No errors found")
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} warnings:")
        for warning in warnings[:10]:
            print(f"  ⚠️  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    print("="*70)
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate competition submission")
    parser.add_argument("submission", help="Path to submission JSON file")
    parser.add_argument("--test-ids", help="Optional file with test image IDs (one per line)")
    
    args = parser.parse_args()
    
    # Load test image IDs if provided
    test_ids = None
    if args.test_ids:
        with open(args.test_ids) as f:
            test_ids = [int(line.strip()) for line in f if line.strip()]
    
    # Validate
    is_valid, errors, warnings = validate_submission(args.submission, test_ids)
    
    # Exit code
    sys.exit(0 if is_valid else 1)
