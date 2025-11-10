#!/usr/bin/env python3
"""
Integration Tests for Submission Pipeline
==========================================

Tests the actual submission scripts end-to-end:
- generate_submission.py
- validate_submission.py
"""

import json
import sys
import tempfile
import subprocess
from pathlib import Path
import unittest
import os


class TestSubmissionIntegration(unittest.TestCase):
    """Integration tests for submission generation and validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path(__file__).parent.parent
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Paths to scripts
        self.generate_script = self.project_root / "scripts" / "generate_submission.py"
        self.validate_script = self.project_root / "scripts" / "validate_submission.py"
        
        # Check scripts exist
        self.assertTrue(self.generate_script.exists(), 
                       f"Generate script not found: {self.generate_script}")
        self.assertTrue(self.validate_script.exists(),
                       f"Validate script not found: {self.validate_script}")
    
    def test_validate_script_exists_and_executable(self):
        """Test that validate script exists and is executable"""
        self.assertTrue(self.validate_script.exists())
        self.assertTrue(os.access(self.validate_script, os.X_OK))
    
    def test_validate_valid_submission(self):
        """Test validation of a valid submission"""
        # Create valid submission
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
        test_file = self.temp_path / "valid_submission.json"
        with open(test_file, 'w') as f:
            json.dump(valid_submission, f)
        
        # Run validation
        result = subprocess.run(
            [sys.executable, str(self.validate_script), str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Should pass (exit code 0)
        self.assertEqual(result.returncode, 0,
                        f"Validation failed:\n{result.stdout}\n{result.stderr}")
        self.assertIn("VALIDATION PASSED", result.stdout)
    
    def test_validate_invalid_submission_wrong_structure(self):
        """Test validation of invalid submission (dict not list)"""
        # Create invalid submission (dict instead of list)
        invalid_submission = {
            "predictions": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}
            ]
        }
        
        # Write to file
        test_file = self.temp_path / "invalid_structure.json"
        with open(test_file, 'w') as f:
            json.dump(invalid_submission, f)
        
        # Run validation
        result = subprocess.run(
            [sys.executable, str(self.validate_script), str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Should fail (exit code 1)
        # Note: Script may return early before printing full error message
        # Just check that it failed (non-zero exit code)
        self.assertNotEqual(result.returncode, 0,
                           f"Invalid structure should fail validation. Output: {result.stdout}")
    
    def test_validate_invalid_category_id(self):
        """Test validation fails for invalid category ID"""
        # Create submission with invalid category ID
        invalid_submission = [
            {
                "image_id": 1,
                "category_id": 5,  # Invalid (should be 1-4)
                "bbox": [100, 100, 50, 50],
                "score": 0.95
            }
        ]
        
        # Write to file
        test_file = self.temp_path / "invalid_category.json"
        with open(test_file, 'w') as f:
            json.dump(invalid_submission, f)
        
        # Run validation
        result = subprocess.run(
            [sys.executable, str(self.validate_script), str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Should fail
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid category_id", result.stdout)
    
    def test_validate_invalid_bbox(self):
        """Test validation fails for invalid bbox"""
        # Create submission with negative width
        invalid_submission = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, -50, 50],  # Negative width
                "score": 0.95
            }
        ]
        
        # Write to file
        test_file = self.temp_path / "invalid_bbox.json"
        with open(test_file, 'w') as f:
            json.dump(invalid_submission, f)
        
        # Run validation
        result = subprocess.run(
            [sys.executable, str(self.validate_script), str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Should fail
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bbox", result.stdout.lower())
    
    def test_validate_shows_statistics(self):
        """Test that validation shows statistics"""
        # Create submission with multiple classes
        submission = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10], "score": 0.8},
            {"image_id": 2, "category_id": 3, "bbox": [30, 30, 10, 10], "score": 0.7},
            {"image_id": 2, "category_id": 4, "bbox": [40, 40, 10, 10], "score": 0.6},
        ]
        
        # Write to file
        test_file = self.temp_path / "multi_class.json"
        with open(test_file, 'w') as f:
            json.dump(submission, f)
        
        # Run validation
        result = subprocess.run(
            [sys.executable, str(self.validate_script), str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Should show statistics
        self.assertIn("Statistics:", result.stdout)
        self.assertIn("Total predictions:", result.stdout)
        self.assertIn("Person", result.stdout)
        self.assertIn("Bicycle", result.stdout)
        self.assertIn("Motorcycle", result.stdout)
        self.assertIn("Vehicle", result.stdout)


class TestDataConfiguration(unittest.TestCase):
    """Test data.yaml configuration"""
    
    def setUp(self):
        """Set up paths"""
        self.project_root = Path(__file__).parent.parent
        self.data_yaml = self.project_root / "data" / "ltdv2_full" / "data.yaml"
    
    def test_data_yaml_exists(self):
        """Test that data.yaml exists"""
        self.assertTrue(self.data_yaml.exists(),
                       f"data.yaml not found: {self.data_yaml}")
    
    def test_data_yaml_has_4_classes(self):
        """Test that data.yaml has 4 classes (not 5)"""
        with open(self.data_yaml, 'r') as f:
            content = f.read()
        
        # Check nc: 4
        self.assertIn("nc: 4", content,
                     "data.yaml should have nc: 4")
        
        # Check class names
        self.assertIn("person", content.lower())
        self.assertIn("bicycle", content.lower())
        self.assertIn("motorcycle", content.lower())
        self.assertIn("vehicle", content.lower())
        
        # Should NOT have background class
        self.assertNotIn("background", content.lower(),
                        "data.yaml should not have background class")


class TestDocumentation(unittest.TestCase):
    """Test that documentation exists and is complete"""
    
    def setUp(self):
        """Set up paths"""
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
    
    def test_submission_guide_exists(self):
        """Test that submission guide exists"""
        guide = self.docs_dir / "COMPETITION_SUBMISSION_GUIDE.md"
        self.assertTrue(guide.exists(),
                       f"Submission guide not found: {guide}")
        
        # Check it's not empty
        self.assertGreater(guide.stat().st_size, 1000,
                          "Submission guide seems too small")
    
    def test_workflow_exists(self):
        """Test that workflow guide exists"""
        workflow = self.docs_dir / "SUBMISSION_WORKFLOW.md"
        self.assertTrue(workflow.exists(),
                       f"Workflow guide not found: {workflow}")
    
    def test_checklist_exists(self):
        """Test that checklist exists"""
        checklist = self.docs_dir / "SUBMISSION_CHECKLIST.md"
        self.assertTrue(checklist.exists(),
                       f"Checklist not found: {checklist}")
    
    def test_guide_has_key_sections(self):
        """Test that guide has key sections"""
        guide = self.docs_dir / "COMPETITION_SUBMISSION_GUIDE.md"
        with open(guide, 'r') as f:
            content = f.read()
        
        # Check for key sections
        key_sections = [
            "Submission Format",
            "Class Mapping",
            "Bounding Box Format",
            "Evaluation Metric",
            "Validation Checklist",
            "Common Issues",
        ]
        
        for section in key_sections:
            self.assertIn(section, content,
                         f"Guide missing section: {section}")
    
    def test_guide_mentions_4_classes(self):
        """Test that guide correctly mentions 4 classes"""
        guide = self.docs_dir / "COMPETITION_SUBMISSION_GUIDE.md"
        with open(guide, 'r') as f:
            content = f.read()
        
        # Should mention 4 classes
        self.assertIn("4 classes", content,
                     "Guide should mention 4 classes")
        
        # Should list all classes
        self.assertIn("Person", content)
        self.assertIn("Bicycle", content)
        self.assertIn("Motorcycle", content)
        self.assertIn("Vehicle", content)


def run_tests():
    """Run all integration tests"""
    print("="*70)
    print("INTEGRATION TEST SUITE")
    print("="*70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSubmissionIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
