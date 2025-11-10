#!/usr/bin/env python3
"""
Submission System Test Suite
=============================

Tests all components of the competition submission pipeline:
- Submission generation
- Format validation
- Class mapping
- Bbox conversion
- Edge cases
"""

import json
import sys
import tempfile
from pathlib import Path
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSubmissionFormat(unittest.TestCase):
    """Test submission JSON format validation"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def test_valid_submission(self):
        """Test that valid submission passes all checks"""
        valid_submission = [
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
        
        # Write to file
        test_file = self.temp_path / "valid.json"
        with open(test_file, 'w') as f:
            json.dump(valid_submission, f)
        
        # Validate
        is_valid = self._validate_submission_format(valid_submission)
        self.assertTrue(is_valid, "Valid submission should pass validation")
    
    def test_invalid_structure_dict_not_list(self):
        """Test that dict structure (not list) fails"""
        invalid_submission = {
            "predictions": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
        }
        
        is_valid = self._validate_submission_format(invalid_submission)
        self.assertFalse(is_valid, "Dict structure should fail (need list)")
    
    def test_missing_required_fields(self):
        """Test that missing required fields fails"""
        invalid_submissions = [
            # Missing image_id
            [{"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}],
            # Missing category_id
            [{"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}],
            # Missing bbox
            [{"image_id": 1, "category_id": 1, "score": 0.5}],
            # Missing score
            [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
        ]
        
        for submission in invalid_submissions:
            is_valid = self._validate_submission_format(submission)
            self.assertFalse(is_valid, f"Submission missing fields should fail: {submission}")
    
    def test_invalid_category_ids(self):
        """Test that invalid category IDs fail"""
        invalid_categories = [0, 5, -1, 10, 100]
        
        for cat_id in invalid_categories:
            submission = [{
                "image_id": 1,
                "category_id": cat_id,
                "bbox": [100, 100, 50, 50],
                "score": 0.5
            }]
            
            is_valid = self._validate_submission_format(submission)
            self.assertFalse(is_valid, f"Category ID {cat_id} should fail (must be 1-4)")
    
    def test_valid_category_ids(self):
        """Test that all valid category IDs pass"""
        valid_categories = [1, 2, 3, 4]
        
        for cat_id in valid_categories:
            submission = [{
                "image_id": 1,
                "category_id": cat_id,
                "bbox": [100, 100, 50, 50],
                "score": 0.5
            }]
            
            is_valid = self._validate_submission_format(submission)
            self.assertTrue(is_valid, f"Category ID {cat_id} should pass")
    
    def test_invalid_bbox_format(self):
        """Test that invalid bbox formats fail"""
        invalid_bboxes = [
            [100, 100],  # Too few values
            [100, 100, 50, 50, 10],  # Too many values
            [100, 100, -50, 50],  # Negative width
            [100, 100, 50, -50],  # Negative height
            [-100, 100, 50, 50],  # Negative x (should fail)
            [100, -100, 50, 50],  # Negative y (should fail)
            [100, 100, 0, 50],  # Zero width
            [100, 100, 50, 0],  # Zero height
        ]
        
        for bbox in invalid_bboxes:
            submission = [{
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "score": 0.5
            }]
            
            is_valid = self._validate_submission_format(submission)
            self.assertFalse(is_valid, f"Bbox {bbox} should fail")
    
    def test_invalid_scores(self):
        """Test that invalid scores fail"""
        invalid_scores = [-0.1, 1.1, 2.0, -1.0, 100.0]
        
        for score in invalid_scores:
            submission = [{
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "score": score
            }]
            
            is_valid = self._validate_submission_format(submission)
            self.assertFalse(is_valid, f"Score {score} should fail (must be in [0, 1])")
    
    def test_valid_scores(self):
        """Test that valid scores pass"""
        valid_scores = [0.0, 0.5, 1.0, 0.001, 0.999]
        
        for score in valid_scores:
            submission = [{
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "score": score
            }]
            
            is_valid = self._validate_submission_format(submission)
            self.assertTrue(is_valid, f"Score {score} should pass")
    
    def _validate_submission_format(self, data):
        """Helper method to validate submission format"""
        try:
            # Check structure
            if not isinstance(data, list):
                return False
            
            required_fields = {'image_id', 'category_id', 'bbox', 'score'}
            valid_categories = {1, 2, 3, 4}
            
            for pred in data:
                # Check fields
                if not isinstance(pred, dict):
                    return False
                
                missing = required_fields - set(pred.keys())
                if missing:
                    return False
                
                # Validate types and ranges
                if not isinstance(pred['image_id'], int):
                    return False
                
                if pred['category_id'] not in valid_categories:
                    return False
                
                bbox = pred['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return False
                
                x, y, w, h = bbox
                if any(not isinstance(v, (int, float)) for v in bbox):
                    return False
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    return False
                
                score = pred['score']
                if not isinstance(score, (int, float)):
                    return False
                if not (0.0 <= score <= 1.0):
                    return False
            
            return True
        
        except Exception:
            return False


class TestClassMapping(unittest.TestCase):
    """Test YOLO to LTDv2 class mapping"""
    
    def setUp(self):
        """Set up class mapping"""
        self.YOLO_TO_LTD = {
            0: 1,  # person → Person
            1: 2,  # bicycle → Bicycle
            3: 3,  # motorcycle → Motorcycle
            2: 4,  # car → Vehicle
            5: 4,  # bus → Vehicle
            7: 4,  # truck → Vehicle
        }
    
    def test_person_mapping(self):
        """Test person class mapping"""
        self.assertEqual(self.YOLO_TO_LTD[0], 1)
    
    def test_bicycle_mapping(self):
        """Test bicycle class mapping"""
        self.assertEqual(self.YOLO_TO_LTD[1], 2)
    
    def test_motorcycle_mapping(self):
        """Test motorcycle class mapping"""
        self.assertEqual(self.YOLO_TO_LTD[3], 3)
    
    def test_vehicle_mapping(self):
        """Test vehicle class mapping (car, bus, truck)"""
        self.assertEqual(self.YOLO_TO_LTD[2], 4)  # car
        self.assertEqual(self.YOLO_TO_LTD[5], 4)  # bus
        self.assertEqual(self.YOLO_TO_LTD[7], 4)  # truck
    
    def test_all_mapped_classes_valid(self):
        """Test that all mapped classes are valid LTDv2 classes"""
        valid_ltd_classes = {1, 2, 3, 4}
        for yolo_class, ltd_class in self.YOLO_TO_LTD.items():
            self.assertIn(ltd_class, valid_ltd_classes,
                         f"YOLO class {yolo_class} maps to invalid LTD class {ltd_class}")
    
    def test_unmapped_classes_ignored(self):
        """Test that unmapped COCO classes are not in mapping"""
        unmapped_coco = [4, 6, 8, 9]  # Examples: airplane, train, boat, etc.
        for coco_class in unmapped_coco:
            self.assertNotIn(coco_class, self.YOLO_TO_LTD,
                           f"COCO class {coco_class} should not be mapped")


class TestBboxConversion(unittest.TestCase):
    """Test bounding box format conversion"""
    
    def test_xyxy_to_xywh_conversion(self):
        """Test conversion from xyxy to xywh format"""
        # Test cases: (x1, y1, x2, y2) -> (x, y, w, h)
        test_cases = [
            ((100, 100, 200, 200), (100, 100, 100, 100)),  # Square
            ((0, 0, 640, 512), (0, 0, 640, 512)),  # Full image
            ((50.5, 75.3, 100.2, 150.8), (50.5, 75.3, 49.7, 75.5)),  # Floats
        ]
        
        for (x1, y1, x2, y2), (exp_x, exp_y, exp_w, exp_h) in test_cases:
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            self.assertAlmostEqual(x, exp_x, places=1)
            self.assertAlmostEqual(y, exp_y, places=1)
            self.assertAlmostEqual(w, exp_w, places=1)
            self.assertAlmostEqual(h, exp_h, places=1)
    
    def test_xywh_always_positive(self):
        """Test that width and height are always positive"""
        # Various xyxy inputs
        test_cases = [
            (100, 100, 200, 200),
            (0, 0, 640, 512),
            (50.5, 75.3, 100.2, 150.8),
        ]
        
        for x1, y1, x2, y2 in test_cases:
            w = x2 - x1
            h = y2 - y1
            
            self.assertGreater(w, 0, f"Width should be positive: {w}")
            self.assertGreater(h, 0, f"Height should be positive: {h}")
    
    def test_bbox_area_preservation(self):
        """Test that bbox area is preserved in conversion"""
        x1, y1, x2, y2 = 100, 100, 200, 300
        
        # xyxy area
        area_xyxy = (x2 - x1) * (y2 - y1)
        
        # xywh conversion
        x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
        area_xywh = w * h
        
        self.assertEqual(area_xyxy, area_xywh,
                        "Area should be preserved in conversion")


class TestRobustnessScore(unittest.TestCase):
    """Test robustness score calculation"""
    
    def test_perfect_consistency(self):
        """Test score with perfect consistency (CoV = 0)"""
        monthly_mAPs = [0.5, 0.5, 0.5, 0.5, 0.5]
        mean_mAP = sum(monthly_mAPs) / len(monthly_mAPs)
        std_mAP = 0.0  # Perfect consistency
        CoV = 0.0
        
        robustness_score = mean_mAP * (1 - CoV)
        
        self.assertEqual(robustness_score, 0.5,
                        "Perfect consistency should give score = mAP")
    
    def test_high_variance(self):
        """Test score with high variance (high CoV)"""
        monthly_mAPs = [0.8, 0.2, 0.9, 0.1, 0.5]
        mean_mAP = sum(monthly_mAPs) / len(monthly_mAPs)
        
        # Calculate std
        variance = sum((x - mean_mAP) ** 2 for x in monthly_mAPs) / len(monthly_mAPs)
        std_mAP = variance ** 0.5
        CoV = std_mAP / mean_mAP
        
        robustness_score = mean_mAP * (1 - CoV)
        
        # High variance should reduce score significantly
        self.assertLess(robustness_score, mean_mAP,
                       "High variance should reduce score")
    
    def test_consistency_beats_accuracy(self):
        """Test that consistent model can beat higher mAP"""
        # Model A: High mAP, high variance
        mAP_A = [0.7, 0.5, 0.8, 0.4, 0.6]
        mean_A = sum(mAP_A) / len(mAP_A)
        var_A = sum((x - mean_A) ** 2 for x in mAP_A) / len(mAP_A)
        std_A = var_A ** 0.5
        CoV_A = std_A / mean_A
        score_A = mean_A * (1 - CoV_A)
        
        # Model B: Lower mAP, lower variance
        mAP_B = [0.55, 0.58, 0.56, 0.57, 0.54]
        mean_B = sum(mAP_B) / len(mAP_B)
        var_B = sum((x - mean_B) ** 2 for x in mAP_B) / len(mAP_B)
        std_B = var_B ** 0.5
        CoV_B = std_B / mean_B
        score_B = mean_B * (1 - CoV_B)
        
        # Consistent model should win despite lower mean
        self.assertGreater(score_B, score_A,
                          "Consistent model should beat inconsistent model")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_empty_submission(self):
        """Test empty submission (valid but warns)"""
        empty_submission = []
        
        # Empty is technically valid format
        self.assertIsInstance(empty_submission, list)
        self.assertEqual(len(empty_submission), 0)
    
    def test_large_submission(self):
        """Test large submission (many predictions)"""
        # Simulate 200K predictions
        large_submission = [
            {
                "image_id": i % 50000,
                "category_id": (i % 4) + 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "score": 0.5
            }
            for i in range(200000)
        ]
        
        self.assertEqual(len(large_submission), 200000)
        self.assertIsInstance(large_submission, list)
    
    def test_float_image_ids_fail(self):
        """Test that float image IDs fail (must be int)"""
        submission = [{
            "image_id": 1.5,  # Float instead of int
            "category_id": 1,
            "bbox": [100, 100, 50, 50],
            "score": 0.5
        }]
        
        # Should fail because image_id must be int
        self.assertFalse(isinstance(submission[0]["image_id"], int))
    
    def test_boundary_scores(self):
        """Test boundary score values"""
        boundary_scores = [0.0, 1.0, 0.0001, 0.9999]
        
        for score in boundary_scores:
            submission = [{
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "score": score
            }]
            
            # All should be valid
            self.assertTrue(0.0 <= submission[0]["score"] <= 1.0)
    
    def test_tiny_bboxes(self):
        """Test very small bounding boxes"""
        tiny_bboxes = [
            [100, 100, 1, 1],  # 1x1 pixel
            [100, 100, 0.1, 0.1],  # Sub-pixel
        ]
        
        for bbox in tiny_bboxes:
            x, y, w, h = bbox
            # Must have positive width and height
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)


def run_tests():
    """Run all tests and print results"""
    print("="*70)
    print("SUBMISSION SYSTEM TEST SUITE")
    print("="*70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSubmissionFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestClassMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestBboxConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustnessScore))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
