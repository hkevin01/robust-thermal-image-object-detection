"""
Unit tests for LTDv2Dataset.

Tests data loading, annotation parsing, error handling, and edge cases.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import LTDv2Dataset


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create image directory
        img_dir = tmpdir / "images"
        img_dir.mkdir()

        # Create sample images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            img.save(img_dir / f"image_{i}.png")

        # Create CSV annotations
        annotations = []
        for i in range(5):
            annotations.append(
                {
                    "image_id": f"image_{i}",
                    "bbox": [10, 10, 100, 100],
                    "category": "Person",
                }
            )

        ann_file = tmpdir / "annotations.csv"
        df = pd.DataFrame(annotations)
        df.to_csv(ann_file, index=False)

        # Create COCO JSON annotations
        coco_data = {
            "images": [{"id": i, "file_name": f"image_{i}.png"} for i in range(5)],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [10, 10, 90, 90],
                    "area": 8100,
                    "iscrowd": 0,
                }
                for i in range(5)
            ],
            "categories": [
                {"id": 1, "name": "Person"},
                {"id": 2, "name": "Bicycle"},
                {"id": 3, "name": "Motorcycle"},
                {"id": 4, "name": "Vehicle"},
            ],
        }

        coco_file = tmpdir / "annotations_coco.json"
        with open(coco_file, "w") as f:
            json.dump(coco_data, f)

        # Create metadata
        metadata = []
        for i in range(5):
            metadata.append(
                {
                    "image_id": f"image_{i}",
                    "temperature": 20.0 + i,
                    "humidity": 50.0 + i * 2,
                    "solar_radiation": 500.0 + i * 10,
                }
            )

        meta_file = tmpdir / "metadata.csv"
        pd.DataFrame(metadata).to_csv(meta_file, index=False)

        yield {
            "dir": tmpdir,
            "images": img_dir,
            "annotations_csv": ann_file,
            "annotations_coco": coco_file,
            "metadata": meta_file,
        }


class TestLTDv2Dataset:
    """Test suite for LTDv2Dataset."""

    def test_init_csv(self, temp_dataset_dir):
        """Test dataset initialization with CSV annotations."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
        )

        assert len(dataset) == 5
        assert dataset.NUM_CLASSES == 4

    def test_init_coco_json(self, temp_dataset_dir):
        """Test dataset initialization with COCO JSON annotations."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_coco"]),
        )

        assert len(dataset) == 5

    def test_init_with_metadata(self, temp_dataset_dir):
        """Test dataset initialization with metadata."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
            metadata_file=str(temp_dataset_dir["metadata"]),
        )

        assert dataset.metadata is not None
        assert len(dataset.metadata) == 5

    def test_getitem(self, temp_dataset_dir):
        """Test __getitem__ returns correct format."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
        )

        sample = dataset[0]

        assert "image" in sample
        assert "boxes" in sample
        assert "labels" in sample
        assert "image_id" in sample

        # Check types
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["boxes"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)

        # Check shapes
        assert sample["image"].dim() == 3  # (C, H, W)
        assert sample["boxes"].dim() == 2  # (N, 4)
        assert sample["labels"].dim() == 1  # (N,)

    def test_getitem_with_metadata(self, temp_dataset_dir):
        """Test __getitem__ returns metadata when available."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
            metadata_file=str(temp_dataset_dir["metadata"]),
        )

        sample = dataset[0]

        assert "metadata" in sample
        assert sample["metadata"] is not None
        assert "temperature" in sample["metadata"]

    def test_missing_image_dir(self):
        """Test error handling for missing image directory."""
        with pytest.raises(ValueError, match="Image directory does not exist"):
            LTDv2Dataset(
                image_dir="/nonexistent/path", annotation_file="/fake/annotations.csv"
            )

    def test_missing_annotation_file(self, temp_dataset_dir):
        """Test error handling for missing annotation file."""
        with pytest.raises(ValueError, match="Annotation file does not exist"):
            LTDv2Dataset(
                image_dir=str(temp_dataset_dir["images"]),
                annotation_file="/nonexistent/annotations.csv",
            )

    def test_corrupted_image_handling(self, temp_dataset_dir):
        """Test handling of corrupted images with retry."""
        # Create a corrupted image file
        corrupted_img = temp_dataset_dir["images"] / "corrupted.png"
        with open(corrupted_img, "wb") as f:
            f.write(b"not an image")

        # Add annotation for corrupted image
        ann_file = temp_dataset_dir["dir"] / "ann_corrupted.csv"
        df = pd.DataFrame(
            [{"image_id": "corrupted", "bbox": [10, 10, 100, 100], "category": "Person"}]
        )
        df.to_csv(ann_file, index=False)

        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(ann_file),
            max_retries=2,
        )

        # Should raise RuntimeError after retries
        with pytest.raises(RuntimeError, match="Failed to load image"):
            _ = dataset[0]

    def test_bbox_normalization(self, temp_dataset_dir):
        """Test bounding box format normalization."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
        )

        sample = dataset[0]
        boxes = sample["boxes"]

        # Check boxes are in [x1, y1, x2, y2] format
        assert torch.all(boxes[:, 2] >= boxes[:, 0])  # x2 >= x1
        assert torch.all(boxes[:, 3] >= boxes[:, 1])  # y2 >= y1

    def test_label_range(self, temp_dataset_dir):
        """Test labels are within valid range."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            labels = sample["labels"]

            assert torch.all(labels >= 0)
            assert torch.all(labels < dataset.NUM_CLASSES)

    def test_statistics(self, temp_dataset_dir):
        """Test get_statistics returns valid info."""
        dataset = LTDv2Dataset(
            image_dir=str(temp_dataset_dir["images"]),
            annotation_file=str(temp_dataset_dir["annotations_csv"]),
        )

        # Load a few samples to generate stats
        for i in range(3):
            _ = dataset[i]

        stats = dataset.get_statistics()

        assert "total_samples" in stats
        assert "load_failures" in stats
        assert "avg_load_time_ms" in stats
        assert stats["total_samples"] == 5

    def test_empty_annotations(self, temp_dataset_dir):
        """Test handling of empty annotation file."""
        empty_ann = temp_dataset_dir["dir"] / "empty.csv"
        pd.DataFrame().to_csv(empty_ann, index=False)

        with pytest.raises(RuntimeError, match="Annotation file is empty"):
            LTDv2Dataset(
                image_dir=str(temp_dataset_dir["images"]), annotation_file=str(empty_ann)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
