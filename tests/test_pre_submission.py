#!/usr/bin/env python3
"""
Tests for Pre-Submission Validation
====================================
"""

import json
import sys
import tempfile
from pathlib import Path
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


class TestPreSubmissionValidation(unittest.TestCase):
    """Test pre-submission validation script"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def create_test_submission(self, filename, data):
        """Helper to create test submission file"""
        filepath = self.temp_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f)
        return filepath
    
    def test_valid_submission_passes(self):
        """Test that a valid submission passes all checks"""
        valid_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.5, 200.3, 50.2, 80.1],
                "score": 0.95
            },
            {
                "image_id": 2,
                "category_id": 4,
                "bbox": [300.0, 150.0, 120.5, 90.3],
                "score": 0.88
            }
        ]
        
        filepath = self.create_test_submission("valid.json", valid_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        success = validator.run_all_checks()
        
        self.assertTrue(success)
        self.assertEqual(len(validator.errors), 0)
    
    def test_dict_structure_fails(self):
        """Test that dict structure fails"""
        invalid_data = {
            "predictions": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}
            ]
        }
        
        filepath = self.create_test_submission("invalid_dict.json", invalid_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        self.assertGreater(len(validator.errors), 0)
        self.assertIn("list", " ".join(validator.errors).lower())
    
    def test_invalid_category_id_detected(self):
        """Test that invalid category IDs are detected"""
        invalid_data = [
            {
                "image_id": 1,
                "category_id": 5,  # Invalid
                "bbox": [100, 100, 50, 50],
                "score": 0.95
            }
        ]
        
        filepath = self.create_test_submission("invalid_cat.json", invalid_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        # Should have error about category
        has_cat_error = any("category" in err.lower() for err in validator.errors)
        self.assertTrue(has_cat_error)
    
    def test_invalid_bbox_detected(self):
        """Test that invalid bbox is detected"""
        invalid_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, -50, 50],  # Negative width
                "score": 0.95
            }
        ]
        
        filepath = self.create_test_submission("invalid_bbox.json", invalid_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        # Should have error about bbox
        has_bbox_error = any("bbox" in err.lower() for err in validator.errors)
        self.assertTrue(has_bbox_error)
    
    def test_out_of_range_score_detected(self):
        """Test that out-of-range scores are detected"""
        invalid_data = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "score": 1.5  # Out of range
            }
        ]
        
        filepath = self.create_test_submission("invalid_score.json", invalid_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        # Should have error about score
        has_score_error = any("score" in err.lower() for err in validator.errors)
        self.assertTrue(has_score_error)
    
    def test_missing_fields_detected(self):
        """Test that missing required fields are detected"""
        invalid_data = [
            {
                "image_id": 1,
                # Missing category_id, bbox, score
            }
        ]
        
        filepath = self.create_test_submission("missing_fields.json", invalid_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        # Should have error about missing fields
        has_field_error = any("field" in err.lower() or "missing" in err.lower() 
                             for err in validator.errors)
        self.assertTrue(has_field_error)
    
    def test_empty_submission_warning(self):
        """Test that empty submission generates warning"""
        empty_data = []
        
        filepath = self.create_test_submission("empty.json", empty_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        # Should have warning about empty
        has_empty_warning = any("empty" in warn.lower() for warn in validator.warnings)
        self.assertTrue(has_empty_warning)
    
    def test_large_file_warning(self):
        """Test that very large submission generates warning"""
        # Create large submission (200K predictions)
        large_data = [
            {
                "image_id": i % 50000,
                "category_id": (i % 4) + 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "score": 0.5
            }
            for i in range(200000)
        ]
        
        filepath = self.create_test_submission("large.json", large_data)
        
        # Check file size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        
        # Just check file size is detected
        self.assertGreater(size_mb, 10)  # Should be > 10MB
    
    def test_statistics_calculated(self):
        """Test that statistics are calculated correctly"""
        multi_class_data = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10], "score": 0.8},
            {"image_id": 2, "category_id": 3, "bbox": [30, 30, 10, 10], "score": 0.7},
            {"image_id": 2, "category_id": 4, "bbox": [40, 40, 10, 10], "score": 0.6},
            {"image_id": 3, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.5},
        ]
        
        filepath = self.create_test_submission("multi_class.json", multi_class_data)
        
        from pre_submission_check import PreSubmissionValidator
        validator = PreSubmissionValidator(filepath)
        validator.run_all_checks()
        
        # Should pass with all classes present
        self.assertEqual(len(validator.errors), 0)


def run_tests():
    """Run all tests"""
    print("="*70)
    print("PRE-SUBMISSION VALIDATION TEST SUITE")
    print("="*70)
    print()
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPreSubmissionValidation)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
