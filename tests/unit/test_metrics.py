"""
Unit tests for TemporalDetectionMetrics.

Tests metric computation, temporal consistency, and challenge scoring.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import TemporalDetectionMetrics


@pytest.fixture
def sample_predictions():
    """Create sample predictions and targets."""
    # Predictions: list of dicts with boxes, scores, labels
    predictions = [
        {
            "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([[15, 15, 55, 55]]),
            "scores": torch.tensor([0.85]),
            "labels": torch.tensor([0]),
        },
    ]

    # Targets: list of dicts with boxes, labels
    targets = [
        {
            "boxes": torch.tensor([[12, 12, 52, 52], [62, 62, 102, 102]]),
            "labels": torch.tensor([0, 1]),
        },
        {"boxes": torch.tensor([[16, 16, 56, 56]]), "labels": torch.tensor([0])},
    ]

    return predictions, targets


@pytest.fixture
def temporal_data():
    """Create temporal data spanning multiple months."""
    months = ["2023-01", "2023-02", "2023-03"]

    predictions = []
    targets = []
    month_ids = []

    for month in months:
        for _ in range(3):  # 3 samples per month
            predictions.append(
                {
                    "boxes": torch.tensor([[10, 10, 50, 50]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            )
            targets.append(
                {"boxes": torch.tensor([[12, 12, 52, 52]]), "labels": torch.tensor([0])}
            )
            month_ids.append(month)

    return predictions, targets, month_ids


class TestTemporalDetectionMetrics:
    """Test suite for TemporalDetectionMetrics."""

    def test_init(self):
        """Test metric initialization."""
        metrics = TemporalDetectionMetrics(num_classes=4, iou_threshold=0.5)

        assert metrics.num_classes == 4
        assert metrics.iou_threshold == 0.5

    def test_update(self, sample_predictions):
        """Test update method."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets = sample_predictions

        metrics.update(predictions, targets)

        # Check internal state updated
        assert len(metrics.metric._predictions) > 0
        assert len(metrics.metric._targets) > 0

    def test_compute_global_metrics(self, sample_predictions):
        """Test global metrics computation."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets = sample_predictions

        metrics.update(predictions, targets)
        result = metrics.compute_global_metrics()

        assert "map" in result
        assert "map_50" in result
        assert "map_75" in result
        assert "map_per_class" in result

        # Check metrics are in valid range
        assert 0 <= result["map"] <= 1
        assert 0 <= result["map_50"] <= 1

    def test_compute_monthly_metrics(self, temporal_data):
        """Test monthly metrics computation."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets, month_ids = temporal_data

        # Update with temporal info
        for pred, target, month in zip(predictions, targets, month_ids):
            metrics.update([pred], [target], temporal_ids=[month])

        monthly = metrics.compute_monthly_metrics()

        assert "2023-01" in monthly
        assert "2023-02" in monthly
        assert "2023-03" in monthly

        # Check each month has valid mAP
        for month, map_value in monthly.items():
            assert 0 <= map_value <= 1

    def test_compute_temporal_consistency(self, temporal_data):
        """Test coefficient of variation computation."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets, month_ids = temporal_data

        # Update with temporal info
        for pred, target, month in zip(predictions, targets, month_ids):
            metrics.update([pred], [target], temporal_ids=[month])

        # Compute monthly metrics first
        _ = metrics.compute_monthly_metrics()

        # Compute CoV
        cov = metrics.compute_temporal_consistency()

        assert cov >= 0
        # CoV should be small for consistent predictions
        assert cov < 1.0  # Reasonable upper bound

    def test_compute_challenge_score(self, temporal_data):
        """Test challenge score computation."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets, month_ids = temporal_data

        # Update with temporal info
        for pred, target, month in zip(predictions, targets, month_ids):
            metrics.update([pred], [target], temporal_ids=[month])

        # Compute global metrics
        global_metrics = metrics.compute_global_metrics()

        # Compute monthly metrics
        _ = metrics.compute_monthly_metrics()

        # Compute CoV
        cov = metrics.compute_temporal_consistency()

        # Compute challenge score
        score = metrics.compute_challenge_score()

        # Check score formula: mAP * (1 - CoV)
        expected_score = global_metrics["map_50"] * (1 - cov)
        assert abs(score - expected_score) < 1e-6

        # Score should be positive and <= mAP
        assert 0 <= score <= global_metrics["map_50"]

    def test_reset(self, sample_predictions):
        """Test reset functionality."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets = sample_predictions

        metrics.update(predictions, targets)
        metrics.reset()

        # Check internal state cleared
        assert len(metrics.metric._predictions) == 0
        assert len(metrics.metric._targets) == 0
        assert len(metrics.monthly_metrics) == 0

    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        metrics = TemporalDetectionMetrics(num_classes=4)

        predictions = [
            {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,))}
        ]
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]]), "labels": torch.tensor([0])}]

        # Should not raise error
        metrics.update(predictions, targets)
        result = metrics.compute_global_metrics()

        # mAP should be 0 for no predictions
        assert result["map"] == 0.0

    def test_empty_targets(self):
        """Test handling of empty targets."""
        metrics = TemporalDetectionMetrics(num_classes=4)

        predictions = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [{"boxes": torch.empty((0, 4)), "labels": torch.empty((0,))}]

        # Should not raise error
        metrics.update(predictions, targets)
        result = metrics.compute_global_metrics()

        # mAP should be 0 for no targets
        assert result["map"] == 0.0

    def test_different_iou_thresholds(self, sample_predictions):
        """Test metrics with different IoU thresholds."""
        predictions, targets = sample_predictions

        # Test with different thresholds
        for iou_thresh in [0.3, 0.5, 0.75]:
            metrics = TemporalDetectionMetrics(num_classes=4, iou_threshold=iou_thresh)
            metrics.update(predictions, targets)
            result = metrics.compute_global_metrics()

            # All metrics should be valid
            assert "map" in result
            assert 0 <= result["map"] <= 1

    def test_per_class_metrics(self, sample_predictions):
        """Test per-class metric computation."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets = sample_predictions

        metrics.update(predictions, targets)
        result = metrics.compute_global_metrics()

        per_class = result["map_per_class"]

        # Should have entry for each class
        assert len(per_class) == 4

        # Each class metric should be valid
        for class_map in per_class:
            if class_map != -1:  # -1 for classes not present
                assert 0 <= class_map <= 1

    def test_no_temporal_ids(self, sample_predictions):
        """Test that metrics work without temporal IDs."""
        metrics = TemporalDetectionMetrics(num_classes=4)
        predictions, targets = sample_predictions

        # Update without temporal IDs
        metrics.update(predictions, targets)

        # Should still compute global metrics
        result = metrics.compute_global_metrics()
        assert "map" in result

        # Monthly metrics should be empty
        monthly = metrics.compute_monthly_metrics()
        assert len(monthly) == 0

    def test_cov_with_single_month(self):
        """Test CoV computation with single month (should be 0)."""
        metrics = TemporalDetectionMetrics(num_classes=4)

        predictions = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [{"boxes": torch.tensor([[12, 12, 52, 52]]), "labels": torch.tensor([0])}]

        metrics.update(predictions, targets, temporal_ids=["2023-01"])
        _ = metrics.compute_monthly_metrics()
        cov = metrics.compute_temporal_consistency()

        # CoV with single month should be 0
        assert cov == 0.0

    def test_challenge_score_zero_map(self):
        """Test challenge score when mAP is 0."""
        metrics = TemporalDetectionMetrics(num_classes=4)

        # No predictions, so mAP will be 0
        predictions = [
            {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,))}
        ]
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]]), "labels": torch.tensor([0])}]

        metrics.update(predictions, targets, temporal_ids=["2023-01"])
        _ = metrics.compute_global_metrics()
        _ = metrics.compute_monthly_metrics()
        _ = metrics.compute_temporal_consistency()

        score = metrics.compute_challenge_score()

        # Score should be 0
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
