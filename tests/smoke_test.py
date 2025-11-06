"""
Smoke test for robust thermal image object detection.

Creates a minimal dataset and runs inference to verify basic functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import LTDv2Dataset
from src.models.yolo_detector import ThermalYOLOv8


def create_dummy_dataset(tmpdir: Path):
    """Create a minimal dummy dataset for smoke testing."""
    # Create directories
    img_dir = tmpdir / "images"
    img_dir.mkdir()

    # Create 3 dummy thermal images (grayscale)
    print("Creating dummy thermal images...")
    for i in range(3):
        # Simulate thermal image with random grayscale values
        img_array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L").convert("RGB")
        img.save(img_dir / f"thermal_{i}.png")

    # Create CSV annotations
    print("Creating dummy annotations...")
    annotations = []
    for i in range(3):
        annotations.append(
            {
                "image_id": f"thermal_{i}",
                "bbox": [50, 50, 150, 150],  # x1, y1, x2, y2
                "category": "Person",
            }
        )

    # Save as CSV
    import pandas as pd

    ann_file = tmpdir / "annotations.csv"
    pd.DataFrame(annotations).to_csv(ann_file, index=False)

    # Also create COCO format
    coco_data = {
        "images": [{"id": i, "file_name": f"thermal_{i}.png"} for i in range(3)],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [50, 50, 100, 100],  # x, y, w, h (COCO format)
                "area": 10000,
                "iscrowd": 0,
            }
            for i in range(3)
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

    return img_dir, ann_file, coco_file


def test_dataset_loading(img_dir, ann_file):
    """Test dataset loading."""
    print("\n[1/4] Testing dataset loading...")

    dataset = LTDv2Dataset(image_dir=str(img_dir), annotation_file=str(ann_file))

    assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"
    print(f"✓ Dataset loaded successfully: {len(dataset)} samples")

    # Test __getitem__
    sample = dataset[0]
    assert "image" in sample
    assert "boxes" in sample
    assert "labels" in sample

    print(f"✓ Sample format correct: {sample['image'].shape}")


def test_coco_format(img_dir, coco_file):
    """Test COCO format loading."""
    print("\n[2/4] Testing COCO format loading...")

    dataset = LTDv2Dataset(image_dir=str(img_dir), annotation_file=str(coco_file))

    assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"
    print(f"✓ COCO dataset loaded successfully: {len(dataset)} samples")


def test_model_initialization():
    """Test YOLOv8 model initialization."""
    print("\n[3/4] Testing YOLOv8 model initialization...")

    # Use smallest YOLOv8 model for smoke test
    model = ThermalYOLOv8(model_name="yolov8n.pt")

    assert model.model is not None
    print(f"✓ Model initialized successfully: {model.get_model_info()['model_name']}")


def test_inference(img_dir):
    """Test inference on dummy images."""
    print("\n[4/4] Testing inference...")

    model = ThermalYOLOv8(model_name="yolov8n.pt")

    # Run inference on all dummy images
    image_paths = list(img_dir.glob("*.png"))
    results = model.predict(
        source=[str(p) for p in image_paths], conf=0.25, iou=0.45, save=False, verbose=False
    )

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print(f"✓ Inference completed: {len(results)} predictions")

    # Check result format
    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"  - Image {i}: {len(boxes)} detections")


def main():
    """Run smoke tests."""
    print("=" * 60)
    print("SMOKE TEST: Robust Thermal Image Object Detection")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy dataset
        img_dir, ann_file, coco_file = create_dummy_dataset(tmpdir)

        try:
            # Run tests
            test_dataset_loading(img_dir, ann_file)
            test_coco_format(img_dir, coco_file)
            test_model_initialization()
            test_inference(img_dir)

            print("\n" + "=" * 60)
            print("✓ ALL SMOKE TESTS PASSED")
            print("=" * 60)
            return 0

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"✗ SMOKE TEST FAILED: {e}")
            print("=" * 60)
            import traceback

            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
