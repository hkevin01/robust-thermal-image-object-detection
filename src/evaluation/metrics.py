"""
Evaluation metrics for thermal object detection with temporal consistency.

This module implements mAP@0.5, Coefficient of Variation, and other metrics
specifically designed for the LTDv2 challenge.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)


class TemporalDetectionMetrics:
    """
    Comprehensive metrics for thermal object detection with temporal drift.

    Computes:
    - Global mAP@0.5
    - Per-class AP@0.5
    - Monthly mAP@0.5 scores
    - Coefficient of Variation (CoV) across months
    - Final challenge score: mAP@0.5 × CoV
    """

    def __init__(
        self,
        num_classes: int = 4,
        iou_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of object classes
            iou_threshold: IoU threshold for mAP calculation
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.class_names = class_names or [
            f"class_{i}" for i in range(num_classes)
        ]

        # Initialize metric calculator
        self.map_metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=[iou_threshold],
            class_metrics=True,
        )

        # Storage for monthly metrics
        self.monthly_predictions = {}
        self.monthly_targets = {}

        logger.info(
            f"Initialized metrics for {num_classes} classes "
            f"with IoU threshold {iou_threshold}"
        )

    def update(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        month: Optional[int] = None,
    ) -> None:
        """
        Update metrics with batch of predictions.

        Args:
            predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
            targets: List of target dicts with 'boxes', 'labels'
            month: Optional month identifier for temporal tracking
        """
        try:
            # Validate inputs
            self._validate_inputs(predictions, targets)

            # Update global metrics
            self.map_metric.update(predictions, targets)

            # Store for monthly metrics if month provided
            if month is not None:
                if month not in self.monthly_predictions:
                    self.monthly_predictions[month] = []
                    self.monthly_targets[month] = []

                self.monthly_predictions[month].extend(predictions)
                self.monthly_targets[month].extend(targets)

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            raise

    def compute_global_metrics(self) -> Dict[str, float]:
        """
        Compute global detection metrics.

        Returns:
            Dictionary with metrics including:
            - map_50: mAP@0.5
            - map_per_class: Per-class AP@0.5
            - precision: Precision
            - recall: Recall
        """
        try:
            metrics = self.map_metric.compute()

            # Extract mAP@0.5
            map_50 = metrics["map_50"].item()

            # Per-class AP
            per_class_ap = {}
            if "map_per_class" in metrics:
                per_class = metrics["map_per_class"]
                for i, ap in enumerate(per_class):
                    if i < len(self.class_names):
                        per_class_ap[self.class_names[i]] = ap.item()

            result = {
                "map_50": map_50,
                "precision": metrics.get("precision", torch.tensor(0.0)).item(),
                "recall": metrics.get("recall", torch.tensor(0.0)).item(),
                **{f"ap_{name}": ap for name, ap in per_class_ap.items()},
            }

            logger.info(f"Global mAP@0.5: {map_50:.4f}")
            return result

        except Exception as e:
            logger.error(f"Failed to compute global metrics: {e}")
            return {"map_50": 0.0}

    def compute_monthly_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-month detection metrics.

        Returns:
            Dictionary mapping month to metrics dict
        """
        monthly_results = {}

        for month in sorted(self.monthly_predictions.keys()):
            try:
                # Create temporary metric calculator for this month
                month_metric = MeanAveragePrecision(
                    iou_type="bbox",
                    iou_thresholds=[self.iou_threshold],
                )

                # Update with month's predictions
                month_metric.update(
                    self.monthly_predictions[month],
                    self.monthly_targets[month],
                )

                # Compute metrics
                metrics = month_metric.compute()
                map_50 = metrics["map_50"].item()

                monthly_results[month] = {
                    "map_50": map_50,
                    "num_samples": len(self.monthly_predictions[month]),
                }

                logger.info(f"Month {month} mAP@0.5: {map_50:.4f}")

            except Exception as e:
                logger.warning(f"Failed to compute metrics for month {month}: {e}")
                monthly_results[month] = {"map_50": 0.0, "num_samples": 0}

        return monthly_results

    def compute_temporal_consistency(self) -> Dict[str, float]:
        """
        Compute temporal consistency metrics.

        Returns:
            Dictionary with:
            - coefficient_of_variation: CoV of monthly mAP scores
            - std_dev: Standard deviation of monthly mAP
            - mean: Mean monthly mAP
            - min: Minimum monthly mAP
            - max: Maximum monthly mAP
        """
        if not self.monthly_predictions:
            logger.warning("No monthly data available for temporal consistency")
            return {
                "coefficient_of_variation": 0.0,
                "std_dev": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        # Get monthly mAP scores
        monthly_metrics = self.compute_monthly_metrics()
        monthly_map_scores = [
            m["map_50"] for m in monthly_metrics.values() if m["map_50"] > 0
        ]

        if not monthly_map_scores:
            logger.warning("No valid monthly mAP scores")
            return {
                "coefficient_of_variation": 0.0,
                "std_dev": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        # Calculate statistics
        mean_map = np.mean(monthly_map_scores)
        std_map = np.std(monthly_map_scores)

        # Coefficient of Variation (lower is better)
        # Handle division by zero
        if mean_map > 1e-8:
            cov = std_map / mean_map
        else:
            cov = float("inf")

        result = {
            "coefficient_of_variation": float(cov),
            "std_dev": float(std_map),
            "mean": float(mean_map),
            "min": float(np.min(monthly_map_scores)),
            "max": float(np.max(monthly_map_scores)),
            "range": float(np.max(monthly_map_scores) - np.min(monthly_map_scores)),
        }

        logger.info(f"Coefficient of Variation: {cov:.4f}")
        logger.info(f"Monthly mAP std: {std_map:.4f}, mean: {mean_map:.4f}")

        return result

    def compute_challenge_score(self) -> Dict[str, float]:
        """
        Compute final challenge score.

        Challenge score = Global mAP@0.5 × (1 - CoV)
        Higher is better. Balances accuracy and temporal consistency.

        Returns:
            Dictionary with:
            - challenge_score: Final competition score
            - global_map_50: Global mAP@0.5
            - coefficient_of_variation: Temporal consistency
        """
        global_metrics = self.compute_global_metrics()
        temporal_metrics = self.compute_temporal_consistency()

        global_map = global_metrics["map_50"]
        cov = temporal_metrics["coefficient_of_variation"]

        # Challenge score formulation
        # Option 1: mAP × (1 - CoV) - rewards both high mAP and low variance
        # Option 2: mAP × (1 / (1 + CoV)) - alternative formulation
        # Using Option 1 as stated in challenge description

        if cov == float("inf"):
            challenge_score = 0.0
        else:
            # Ensure CoV contribution is reasonable
            consistency_factor = max(0.0, 1.0 - cov)
            challenge_score = global_map * consistency_factor

        result = {
            "challenge_score": float(challenge_score),
            "global_map_50": float(global_map),
            "coefficient_of_variation": float(cov),
            "consistency_factor": float(consistency_factor)
            if cov != float("inf")
            else 0.0,
        }

        logger.info(
            f"Challenge Score: {challenge_score:.4f} "
            f"(mAP: {global_map:.4f}, CoV: {cov:.4f})"
        )

        return result

    def compute_all_metrics(self) -> Dict[str, any]:
        """
        Compute all metrics at once.

        Returns:
            Comprehensive dictionary with all metrics
        """
        global_metrics = self.compute_global_metrics()
        monthly_metrics = self.compute_monthly_metrics()
        temporal_metrics = self.compute_temporal_consistency()
        challenge_metrics = self.compute_challenge_score()

        return {
            "global": global_metrics,
            "monthly": monthly_metrics,
            "temporal": temporal_metrics,
            "challenge": challenge_metrics,
        }

    def reset(self) -> None:
        """Reset all metrics and stored data."""
        self.map_metric.reset()
        self.monthly_predictions.clear()
        self.monthly_targets.clear()
        logger.info("Metrics reset")

    def _validate_inputs(
        self, predictions: List[Dict], targets: List[Dict]
    ) -> None:
        """
        Validate prediction and target inputs.

        Args:
            predictions: List of prediction dicts
            targets: List of target dicts

        Raises:
            ValueError: If inputs are invalid
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Predictions ({len(predictions)}) and targets ({len(targets)}) "
                "must have same length"
            )

        # Validate each prediction
        for i, pred in enumerate(predictions):
            required_keys = ["boxes", "scores", "labels"]
            missing = set(required_keys) - set(pred.keys())
            if missing:
                raise ValueError(
                    f"Prediction {i} missing required keys: {missing}"
                )

            # Check shapes
            if pred["boxes"].shape[0] != pred["labels"].shape[0]:
                raise ValueError(
                    f"Prediction {i}: boxes and labels shape mismatch"
                )

        # Validate each target
        for i, tgt in enumerate(targets):
            required_keys = ["boxes", "labels"]
            missing = set(required_keys) - set(tgt.keys())
            if missing:
                raise ValueError(f"Target {i} missing required keys: {missing}")

            # Check shapes
            if tgt["boxes"].shape[0] != tgt["labels"].shape[0]:
                raise ValueError(f"Target {i}: boxes and labels shape mismatch")


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
        box2: Tensor of shape (M, 4) in (x1, y1, x2, y2) format

    Returns:
        IoU matrix of shape (N, M)
    """
    # Compute intersection area
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Compute IoU
    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + 1e-8)

    return iou
