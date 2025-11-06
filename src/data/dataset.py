"""
LTDv2 Dataset loader with robust error handling and metadata integration.

This module provides a PyTorch Dataset class for loading thermal imagery
from the LTDv2 dataset with comprehensive error handling, boundary condition
checks, and weather metadata integration.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LTDv2Dataset(Dataset):
    """
    LTDv2 Dataset for thermal object detection with temporal drift.

    Args:
        image_dir: Path to directory containing images
        annotation_file: Path to annotation file (COCO format)
        metadata_file: Optional path to weather metadata CSV
        transform: Optional albumentations transform
        mode: Dataset mode ('train', 'val', 'test')
        max_retries: Maximum number of retries for loading corrupted images
        validate_on_init: Whether to validate all images on initialization

    Raises:
        ValueError: If image_dir or annotation_file doesn't exist
        RuntimeError: If no valid images found
    """

    CLASS_NAMES = ["Person", "Bicycle", "Motorcycle", "Vehicle"]
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        metadata_file: Optional[str] = None,
        transform: Optional[Any] = None,
        mode: str = "train",
        max_retries: int = 3,
        validate_on_init: bool = False,
    ):
        super().__init__()

        # Validate inputs
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.mode = mode
        self.transform = transform
        self.max_retries = max_retries

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

        if not self.annotation_file.exists():
            raise ValueError(f"Annotation file does not exist: {self.annotation_file}")

        # Load annotations
        logger.info(f"Loading annotations from {self.annotation_file}")
        self.annotations = self._load_annotations()

        # Load metadata if provided
        self.metadata = None
        if metadata_file and Path(metadata_file).exists():
            logger.info(f"Loading metadata from {metadata_file}")
            self.metadata = self._load_metadata(metadata_file)

        # Track statistics
        self.load_failures = 0
        self.load_times = []

        # Optional validation
        if validate_on_init:
            self._validate_dataset()

        logger.info(
            f"Initialized LTDv2Dataset with {len(self)} samples in {mode} mode"
        )

    def _load_annotations(self) -> pd.DataFrame:
        """
        Load annotations from file with error handling.
        Supports CSV, JSON, and COCO JSON formats.

        Returns:
            DataFrame with annotations

        Raises:
            RuntimeError: If annotation file is corrupted or empty
        """
        try:
            # Support multiple formats
            if self.annotation_file.suffix == ".csv":
                df = pd.read_csv(self.annotation_file)
            elif self.annotation_file.suffix == ".json":
                # Try to load as COCO format first
                df = self._load_coco_annotations()
                if df is None:
                    # Fallback to regular JSON
                    df = pd.read_json(self.annotation_file)
            else:
                raise ValueError(
                    f"Unsupported annotation format: {self.annotation_file.suffix}"
                )

            if df.empty:
                raise RuntimeError("Annotation file is empty")

            # Validate required columns
            required_cols = ["image_id", "bbox", "category"]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            return df

        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            raise RuntimeError(f"Failed to load annotations: {e}") from e

    def _load_coco_annotations(self) -> Optional[pd.DataFrame]:
        """
        Load annotations in COCO format.

        Returns:
            DataFrame with parsed COCO annotations or None if not COCO format
        """
        try:
            import json

            with open(self.annotation_file, "r") as f:
                coco_data = json.load(f)

            # Check if it's COCO format
            if not all(key in coco_data for key in ["images", "annotations", "categories"]):
                return None

            logger.info("Detected COCO format annotations")

            # Create image_id to filename mapping
            image_map = {img["id"]: img["file_name"] for img in coco_data["images"]}

            # Create category_id to name mapping
            category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

            # Parse annotations
            annotations_list = []
            for ann in coco_data["annotations"]:
                image_id = image_map.get(ann["image_id"], str(ann["image_id"]))
                # Remove file extension from image_id if present
                image_id = Path(image_id).stem

                bbox = ann["bbox"]  # COCO format: [x, y, width, height]
                category_id = ann["category_id"]
                category_name = category_map.get(category_id, f"class_{category_id}")

                annotations_list.append(
                    {
                        "image_id": image_id,
                        "bbox": bbox,
                        "category": category_name,
                        "category_id": category_id,
                        "area": ann.get("area", bbox[2] * bbox[3]),
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )

            df = pd.DataFrame(annotations_list)
            logger.info(f"Loaded {len(df)} COCO annotations")

            return df

        except Exception as e:
            logger.warning(f"Failed to parse as COCO format: {e}")
            return None

    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """
        Load weather metadata with error handling.

        Args:
            metadata_file: Path to metadata CSV file

        Returns:
            DataFrame with metadata
        """
        try:
            metadata = pd.read_csv(metadata_file)

            # Expected columns
            expected_cols = ["image_id", "temperature", "humidity", "solar_radiation"]
            available_cols = [col for col in expected_cols if col in metadata.columns]

            if not available_cols:
                logger.warning(
                    "No expected metadata columns found, metadata will not be used"
                )
                return None

            return metadata[available_cols]

        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None

    def _validate_dataset(self) -> None:
        """
        Validate dataset integrity by checking all images can be loaded.

        Logs warnings for corrupted images but doesn't fail.
        """
        logger.info("Validating dataset integrity...")
        corrupted = []

        for idx in range(len(self)):
            try:
                _ = self[idx]
            except Exception as e:
                corrupted.append((idx, str(e)))

        if corrupted:
            logger.warning(
                f"Found {len(corrupted)} corrupted images out of {len(self)}"
            )
            for idx, error in corrupted[:10]:  # Log first 10
                logger.warning(f"  Image {idx}: {error}")
        else:
            logger.info("All images validated successfully")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a single sample with robust error handling.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - boxes: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
                - labels: Tensor of shape (N,)
                - metadata: Optional dict with weather info
                - image_id: Image identifier

        Raises:
            RuntimeError: If image cannot be loaded after max_retries
        """
        start_time = time.time()

        # Get annotation
        annotation = self.annotations.iloc[idx]
        image_id = annotation["image_id"]

        # Load image with retries
        image = self._load_image_with_retry(image_id)

        # Parse boxes and labels
        boxes, labels = self._parse_annotation(annotation)

        # Get metadata if available
        metadata = self._get_metadata(image_id)

        # Apply transformations
        if self.transform:
            try:
                transformed = self.transform(
                    image=image, bboxes=boxes, labels=labels
                )
                image = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["labels"]
            except Exception as e:
                logger.warning(f"Transform failed for image {image_id}: {e}")
                # Continue with original image/boxes

        # Convert to tensors
        image = self._to_tensor(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Validate shapes
        self._validate_output(image, boxes, labels, image_id)

        # Track loading time
        load_time = time.time() - start_time
        self.load_times.append(load_time)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "metadata": metadata,
            "image_id": image_id,
        }

    def _load_image_with_retry(self, image_id: str) -> np.ndarray:
        """
        Load image with retry mechanism for robustness.

        Args:
            image_id: Image identifier

        Returns:
            Loaded image as numpy array (H, W, C)

        Raises:
            RuntimeError: If image cannot be loaded after max_retries
        """
        image_path = self.image_dir / f"{image_id}.png"

        for attempt in range(self.max_retries):
            try:
                # Try OpenCV first (faster)
                image = cv2.imread(str(image_path))

                if image is None:
                    # Fallback to PIL
                    image = np.array(Image.open(image_path))

                if image is None or image.size == 0:
                    raise ValueError("Empty image")

                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Validate dimensions
                if len(image.shape) not in [2, 3]:
                    raise ValueError(f"Invalid image shape: {image.shape}")

                return image

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed "
                    f"for image {image_id}: {e}"
                )
                if attempt == self.max_retries - 1:
                    self.load_failures += 1
                    raise RuntimeError(
                        f"Failed to load image {image_id} after "
                        f"{self.max_retries} attempts"
                    ) from e
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    def _parse_annotation(
        self, annotation: pd.Series
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Parse bounding boxes and labels from annotation.

        Args:
            annotation: Annotation row

        Returns:
            Tuple of (boxes, labels) where boxes are [[x1, y1, x2, y2], ...]
        """
        # Handle different annotation formats
        boxes = []
        labels = []

        try:
            bbox_data = annotation["bbox"]
            category = annotation["category"]

            # Parse bbox (can be string, list, or dict)
            if isinstance(bbox_data, str):
                bbox_data = eval(bbox_data)

            if isinstance(bbox_data, list):
                # Multiple boxes
                for bbox in bbox_data:
                    boxes.append(self._normalize_bbox(bbox))
            else:
                # Single box
                boxes.append(self._normalize_bbox(bbox_data))

            # Parse category
            if isinstance(category, str):
                labels = [self.CLASS_NAMES.index(category)]
            elif isinstance(category, int):
                labels = [category]
            elif isinstance(category, list):
                labels = category

        except Exception as e:
            logger.warning(f"Failed to parse annotation: {e}")
            # Return empty annotations
            boxes = []
            labels = []

        return boxes, labels

    def _normalize_bbox(self, bbox: Any) -> List[float]:
        """
        Normalize bounding box to [x1, y1, x2, y2] format.

        Args:
            bbox: Bounding box in various formats

        Returns:
            Normalized bbox as [x1, y1, x2, y2]
        """
        if isinstance(bbox, dict):
            # COCO format: {x, y, width, height}
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            return [x, y, x + w, y + h]
        elif len(bbox) == 4:
            # Could be [x, y, w, h] or [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            if x2 < x1 or y2 < y1:
                # Assume it's [x, y, w, h]
                return [x1, y1, x1 + x2, y1 + y2]
            return [x1, y1, x2, y2]
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")

    def _get_metadata(self, image_id: str) -> Optional[Dict[str, float]]:
        """
        Retrieve metadata for image.

        Args:
            image_id: Image identifier

        Returns:
            Dictionary with metadata or None
        """
        if self.metadata is None:
            return None

        try:
            row = self.metadata[self.metadata["image_id"] == image_id]
            if row.empty:
                return None

            return row.iloc[0].to_dict()

        except Exception as e:
            logger.warning(f"Failed to get metadata for {image_id}: {e}")
            return None

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor with proper normalization.

        Args:
            image: Image as numpy array (H, W, C) or (H, W)

        Returns:
            Tensor of shape (C, H, W) normalized to [0, 1]
        """
        # Handle grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image

    def _validate_output(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        image_id: str,
    ) -> None:
        """
        Validate output tensors for boundary conditions.

        Args:
            image: Image tensor
            boxes: Boxes tensor
            labels: Labels tensor
            image_id: Image identifier

        Raises:
            ValueError: If validation fails
        """
        # Check image shape
        if image.dim() != 3:
            raise ValueError(
                f"Invalid image dimensions for {image_id}: {image.shape}"
            )

        # Check boxes
        if boxes.numel() > 0:
            if boxes.dim() != 2 or boxes.shape[1] != 4:
                raise ValueError(
                    f"Invalid boxes shape for {image_id}: {boxes.shape}"
                )

            # Check box coordinates are valid
            if torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(
                boxes[:, 3] <= boxes[:, 1]
            ):
                logger.warning(f"Invalid box coordinates for {image_id}")

        # Check labels
        if labels.numel() > 0:
            if labels.dim() != 1:
                raise ValueError(
                    f"Invalid labels shape for {image_id}: {labels.shape}"
                )

            # Check label range
            if torch.any(labels < 0) or torch.any(labels >= self.NUM_CLASSES):
                raise ValueError(
                    f"Invalid label values for {image_id}: {labels}"
                )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset loading statistics.

        Returns:
            Dictionary with statistics
        """
        avg_load_time = (
            np.mean(self.load_times) if self.load_times else 0.0
        )

        return {
            "total_samples": len(self),
            "load_failures": self.load_failures,
            "avg_load_time_ms": avg_load_time * 1000,
            "mode": self.mode,
        }
