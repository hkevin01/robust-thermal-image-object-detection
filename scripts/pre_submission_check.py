#!/usr/bin/env python3
"""
Pre-Submission Validation Checklist
====================================

Comprehensive validation before uploading submission to competition.

This script performs all necessary checks to ensure:
- Submission file is valid
- Format meets competition requirements
- No common errors present
- System is ready for upload

Run this before EVERY submission!
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class PreSubmissionValidator:
    """Comprehensive pre-submission validation"""
    
    def __init__(self, submission_file):
        self.submission_file = Path(submission_file)
        self.errors = []
        self.warnings = []
        self.info = []
        self.passed_checks = 0
        self.total_checks = 0
    
    def print_header(self):
        """Print validation header"""
        print("=" * 70)
        print(f"{Colors.BOLD}PRE-SUBMISSION VALIDATION CHECKLIST{Colors.NC}")
        print("=" * 70)
        print(f"\nFile: {self.submission_file}")
        print(f"Competition: WACV 2026 RWS Thermal Object Detection")
        print(f"URL: https://www.codabench.org/competitions/10954/")
        print("=" * 70)
        print()
    
    def check(self, name, condition, error_msg=None, warning_msg=None):
        """Generic check with pass/fail tracking"""
        self.total_checks += 1
        
        if condition:
            print(f"{Colors.GREEN}✅ {name}{Colors.NC}")
            self.passed_checks += 1
            return True
        else:
            if error_msg:
                print(f"{Colors.RED}❌ {name}{Colors.NC}")
                self.errors.append(error_msg)
            elif warning_msg:
                print(f"{Colors.YELLOW}⚠️  {name}{Colors.NC}")
                self.warnings.append(warning_msg)
            return False
    
    def run_all_checks(self):
        """Run all validation checks"""
        
        # 1. File existence checks
        print(f"{Colors.BOLD}1. File Checks{Colors.NC}")
        print("-" * 70)
        
        file_exists = self.submission_file.exists()
        self.check(
            "Submission file exists",
            file_exists,
            error_msg=f"File not found: {self.submission_file}"
        )
        
        if not file_exists:
            return False
        
        file_size = self.submission_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        self.check(
            f"File size reasonable ({size_mb:.2f} MB)",
            size_mb < 100,
            warning_msg=f"File is large ({size_mb:.2f} MB). Consider compression if upload fails."
        )
        
        self.check(
            "File extension is .json",
            self.submission_file.suffix.lower() == '.json',
            warning_msg="File should have .json extension"
        )
        
        print()
        
        # 2. JSON format checks
        print(f"{Colors.BOLD}2. JSON Format Checks{Colors.NC}")
        print("-" * 70)
        
        try:
            with open(self.submission_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.check("Valid JSON format", True)
        except json.JSONDecodeError as e:
            self.check("Valid JSON format", False, 
                      error_msg=f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.check("Readable file", False,
                      error_msg=f"Cannot read file: {e}")
            return False
        
        is_list = isinstance(data, list)
        self.check(
            "Root structure is list (not dict)",
            is_list,
            error_msg="Submission must be a JSON array (list), not an object (dict)"
        )
        
        if not is_list:
            return False
        
        self.check(
            f"Non-empty submission ({len(data)} predictions)",
            len(data) > 0,
            warning_msg="Submission is empty (0 predictions)"
        )
        
        print()
        
        # 3. Structure validation
        print(f"{Colors.BOLD}3. Prediction Structure{Colors.NC}")
        print("-" * 70)
        
        required_fields = {'image_id', 'category_id', 'bbox', 'score'}
        
        if len(data) > 0:
            # Check first prediction
            first_pred = data[0]
            has_all_fields = required_fields.issubset(set(first_pred.keys()))
            self.check(
                "All required fields present",
                has_all_fields,
                error_msg=f"Missing fields: {required_fields - set(first_pred.keys())}"
            )
            
            # Check no extra fields (warning only)
            extra_fields = set(first_pred.keys()) - required_fields
            if extra_fields:
                self.warnings.append(f"Extra fields present (will be ignored): {extra_fields}")
                print(f"{Colors.YELLOW}⚠️  Extra fields present: {extra_fields}{Colors.NC}")
        
        print()
        
        # 4. Detailed field validation
        print(f"{Colors.BOLD}4. Field Type & Value Validation{Colors.NC}")
        print("-" * 70)
        
        valid_categories = {1, 2, 3, 4}
        
        type_errors = []
        value_errors = []
        image_ids_seen = set()
        category_counts = defaultdict(int)
        
        # Sample check (first 100 and last 100 predictions)
        sample_indices = list(range(min(100, len(data)))) + \
                        list(range(max(0, len(data) - 100), len(data)))
        
        for idx in sample_indices:
            if idx >= len(data):
                continue
                
            pred = data[idx]
            
            # Check image_id type
            if not isinstance(pred.get('image_id'), int):
                type_errors.append(f"Pred {idx}: image_id must be int, got {type(pred.get('image_id'))}")
            else:
                image_ids_seen.add(pred['image_id'])
            
            # Check category_id
            cat_id = pred.get('category_id')
            if not isinstance(cat_id, int):
                type_errors.append(f"Pred {idx}: category_id must be int, got {type(cat_id)}")
            elif cat_id not in valid_categories:
                value_errors.append(f"Pred {idx}: category_id {cat_id} invalid (must be 1-4)")
            else:
                category_counts[cat_id] += 1
            
            # Check bbox
            bbox = pred.get('bbox')
            if not isinstance(bbox, list):
                type_errors.append(f"Pred {idx}: bbox must be list, got {type(bbox)}")
            elif len(bbox) != 4:
                value_errors.append(f"Pred {idx}: bbox must have 4 values, got {len(bbox)}")
            else:
                x, y, w, h = bbox
                if not all(isinstance(v, (int, float)) for v in bbox):
                    type_errors.append(f"Pred {idx}: bbox values must be numbers")
                elif any(v < 0 for v in [x, y]):
                    value_errors.append(f"Pred {idx}: bbox x,y cannot be negative: {bbox}")
                elif w <= 0 or h <= 0:
                    value_errors.append(f"Pred {idx}: bbox width/height must be positive: {bbox}")
            
            # Check score
            score = pred.get('score')
            if not isinstance(score, (int, float)):
                type_errors.append(f"Pred {idx}: score must be number, got {type(score)}")
            elif not (0.0 <= score <= 1.0):
                value_errors.append(f"Pred {idx}: score {score} out of range [0, 1]")
        
        self.check(
            "All field types correct (sampled)",
            len(type_errors) == 0,
            error_msg=f"Type errors: {len(type_errors)} found\n  " + "\n  ".join(type_errors[:5])
        )
        
        self.check(
            "All field values valid (sampled)",
            len(value_errors) == 0,
            error_msg=f"Value errors: {len(value_errors)} found\n  " + "\n  ".join(value_errors[:5])
        )
        
        print()
        
        # 5. Statistical checks
        print(f"{Colors.BOLD}5. Statistical Analysis{Colors.NC}")
        print("-" * 70)
        
        print(f"📊 Total predictions: {len(data)}")
        print(f"📊 Unique images (sampled): {len(image_ids_seen)}")
        
        if len(image_ids_seen) > 0:
            print(f"📊 Predictions per image: {len(data) / len(image_ids_seen):.1f}")
        
        CLASS_NAMES = {1: "Person", 2: "Bicycle", 3: "Motorcycle", 4: "Vehicle"}
        print("\n📊 Per-class distribution:")
        for cat_id in sorted(valid_categories):
            count = category_counts.get(cat_id, 0)
            print(f"   {CLASS_NAMES[cat_id]}: {count}")
        
        # Check class distribution
        self.check(
            "All 4 classes present",
            len(category_counts) == 4,
            warning_msg=f"Only {len(category_counts)} classes present (expected 4)"
        )
        
        # Check score distribution
        scores = [pred['score'] for pred in data[:1000] if isinstance(pred.get('score'), (int, float))]
        if scores:
            print(f"\n�� Score statistics (first 1000):")
            print(f"   Min: {min(scores):.4f}")
            print(f"   Max: {max(scores):.4f}")
            print(f"   Mean: {sum(scores)/len(scores):.4f}")
            
            low_conf_count = sum(1 for s in scores if s < 0.01)
            high_conf_count = sum(1 for s in scores if s > 0.9)
            
            if low_conf_count > len(scores) * 0.8:
                self.warnings.append(f"Many low-confidence predictions ({low_conf_count}/{len(scores)})")
        
        print()
        
        # 6. Competition-specific checks
        print(f"{Colors.BOLD}6. Competition-Specific Checks{Colors.NC}")
        print("-" * 70)
        
        self.check(
            "Category IDs are 1-4 (not 0-3)",
            all(pred.get('category_id') in valid_categories 
                for pred in data[:100]),
            error_msg="Category IDs must be 1-4 (Person, Bicycle, Motorcycle, Vehicle)"
        )
        
        # Check bbox format is xywh not xyxy
        suspicious_bboxes = []
        for i, pred in enumerate(data[:100]):
            bbox = pred.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                # Heuristic: if w or h > 640 (image size), might be xyxy format
                if w > 640 or h > 512:
                    suspicious_bboxes.append(f"Pred {i}: bbox {bbox} looks like xyxy format")
        
        self.check(
            "Bbox format is [x, y, w, h] (not [x1, y1, x2, y2])",
            len(suspicious_bboxes) == 0,
            warning_msg=f"Some bboxes look suspicious:\n  " + "\n  ".join(suspicious_bboxes[:3])
        )
        
        self.check(
            "Using absolute pixel coordinates (not normalized)",
            all(isinstance(pred.get('bbox', [0])[0], (int, float)) and 
                pred.get('bbox', [0])[0] >= 1
                for pred in data[:100] if len(pred.get('bbox', [])) > 0),
            warning_msg="Coordinates seem normalized (should be absolute pixels)"
        )
        
        print()
        
        # 7. Best practices
        print(f"{Colors.BOLD}7. Best Practices{Colors.NC}")
        print("-" * 70)
        
        if len(data) > 0:
            predictions_per_image = len(data) / len(image_ids_seen) if len(image_ids_seen) > 0 else 0
            
            self.check(
                "Reasonable predictions per image (0.5-20)",
                0.5 <= predictions_per_image <= 20,
                warning_msg=f"Unusual predictions/image ratio: {predictions_per_image:.1f}"
            )
            
            # Check file size vs predictions
            bytes_per_pred = file_size / len(data)
            self.check(
                "Efficient file size",
                50 <= bytes_per_pred <= 500,
                warning_msg=f"Unusual bytes/prediction: {bytes_per_pred:.1f}"
            )
        
        print()
        
        return True
    
    def print_summary(self):
        """Print final validation summary"""
        print("=" * 70)
        print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.NC}")
        print("=" * 70)
        print()
        
        print(f"Total Checks: {self.total_checks}")
        print(f"{Colors.GREEN}Passed: {self.passed_checks}{Colors.NC}")
        
        failed = self.total_checks - self.passed_checks
        if failed > 0:
            print(f"{Colors.RED}Failed: {failed}{Colors.NC}")
        else:
            print("Failed: 0")
        
        print()
        
        if self.errors:
            print(f"{Colors.RED}{Colors.BOLD}ERRORS ({len(self.errors)}):{Colors.NC}")
            for i, error in enumerate(self.errors, 1):
                print(f"{Colors.RED}{i}. {error}{Colors.NC}")
            print()
        
        if self.warnings:
            print(f"{Colors.YELLOW}{Colors.BOLD}WARNINGS ({len(self.warnings)}):{Colors.NC}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{Colors.YELLOW}{i}. {warning}{Colors.NC}")
            print()
        
        print("=" * 70)
        
        if len(self.errors) == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ VALIDATION PASSED{Colors.NC}")
            print()
            print("Submission is ready for upload!")
            print()
            print("Next steps:")
            print("1. Go to: https://www.codabench.org/competitions/10954/")
            print("2. Click 'Participate' → 'Submit'")
            print("3. Upload your submission file")
            print("4. Wait for evaluation (10-30 minutes)")
            print("5. Check leaderboard for results")
            print()
            print("=" * 70)
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ VALIDATION FAILED{Colors.NC}")
            print()
            print("Please fix the errors above before uploading.")
            print()
            print("Common fixes:")
            print("- Check bbox format: [x, y, width, height]")
            print("- Verify category IDs are 1-4")
            print("- Ensure scores are in [0.0, 1.0]")
            print("- Validate JSON structure is a list")
            print()
            print("See docs/COMPETITION_SUBMISSION_GUIDE.md for details.")
            print()
            print("=" * 70)
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pre-submission validation for competition uploads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s submission_dev.json
  %(prog)s submission_final.json --verbose

This tool performs comprehensive validation before uploading to:
https://www.codabench.org/competitions/10954/
        """
    )
    
    parser.add_argument(
        'submission',
        help='Path to submission JSON file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = PreSubmissionValidator(args.submission)
    validator.print_header()
    
    try:
        validator.run_all_checks()
        passed = validator.print_summary()
        
        sys.exit(0 if passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.NC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
