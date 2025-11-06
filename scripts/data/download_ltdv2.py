#!/usr/bin/env python3
"""
Download LTDv2 dataset from HuggingFace.

Dataset structure on HuggingFace:
- data/frames.zip (48 GB): All thermal images (~1M images)
- data/Train.json (1 GB): COCO format training annotations
- data/Valid.json (134 MB): COCO format validation annotations
- data/TestNoLabels.json (33 MB): Test split metadata without labels
"""

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed")
    print("Run: pip install huggingface-hub tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ID = "vapaau/LTDv2"


def download_file(filename: str, output_dir: Path) -> Path:
    """Download a file from HuggingFace dataset."""
    logger.info(f"Downloading {filename}...")
    
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(output_dir / ".cache"),
        resume_download=True,
    )
    
    logger.info(f"Downloaded to: {local_path}")
    return Path(local_path)


def extract_frames(zip_path: Path, output_dir: Path) -> Path:
    """Extract frames from zip file."""
    frames_dir = output_dir / "frames"
    
    if frames_dir.exists() and any(frames_dir.iterdir()):
        logger.info(f"Frames already extracted at {frames_dir}")
        return frames_dir
    
    logger.info(f"Extracting frames from {zip_path}...")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress bar
        members = zip_ref.namelist()
        logger.info(f"Extracting {len(members)} files...")
        
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, frames_dir)
    
    logger.info(f"Extraction complete: {frames_dir}")
    return frames_dir


def convert_coco_to_yolo(coco_json_path: Path, frames_dir: Path, output_dir: Path, split: str):
    """Convert COCO JSON annotations to YOLO format."""
    logger.info(f"Converting {split} annotations to YOLO format...")
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Build image id to filename mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Build category id to class id mapping (0-indexed)
    category_map = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Process each image
    logger.info(f"Processing {len(image_map)} images...")
    for img_id, img_info in tqdm(image_map.items(), desc=f"Converting {split}"):
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Find source image
        source_img = frames_dir / filename
        if not source_img.exists():
            # Try alternative paths
            source_img = frames_dir / "frames" / filename
            if not source_img.exists():
                logger.warning(f"Image not found: {filename}")
                continue
        
        # Copy/symlink image to output
        dest_img = images_dir / Path(filename).name
        if not dest_img.exists():
            shutil.copy2(source_img, dest_img)
        
        # Create YOLO annotation file
        label_file = labels_dir / (Path(filename).stem + '.txt')
        
        if img_id in image_annotations:
            with open(label_file, 'w') as f:
                for ann in image_annotations[img_id]:
                    # Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] (normalized)
                    x, y, w, h = ann['bbox']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    class_id = category_map[ann['category_id']]
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        else:
            # Create empty label file if no annotations
            label_file.touch()
    
    logger.info(f"Converted {split} split: {len(image_map)} images")


def create_dataset_yaml(output_dir: Path, categories: list):
    """Create YOLO dataset.yaml configuration file."""
    yaml_content = f"""# LTDv2 Dataset Configuration
# Thermal object detection dataset with 1M+ images
# Source: https://huggingface.co/datasets/vapaau/LTDv2

path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(categories)}
names:
"""
    
    for idx, cat in enumerate(categories):
        yaml_content += f"  {idx}: {cat['name']}\n"
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created dataset configuration: {yaml_path}")


def download_ltdv2_full(output_dir: Path):
    """Download and prepare the full LTDv2 dataset."""
    logger.info("Starting full LTDv2 dataset download...")
    logger.info("This will download ~49 GB and may take 2-4 hours")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download all files
    logger.info("\n=== Step 1/4: Downloading files ===")
    frames_zip = download_file("data/frames.zip", output_dir)
    train_json = download_file("data/Train.json", output_dir)
    valid_json = download_file("data/Valid.json", output_dir)
    test_json = download_file("data/TestNoLabels.json", output_dir)
    
    # Step 2: Extract frames
    logger.info("\n=== Step 2/4: Extracting frames ===")
    frames_dir = extract_frames(frames_zip, output_dir)
    
    # Step 3: Convert annotations to YOLO format
    logger.info("\n=== Step 3/4: Converting annotations ===")
    
    # Load train json to get categories
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    categories = train_data['categories']
    
    convert_coco_to_yolo(train_json, frames_dir, output_dir, "train")
    convert_coco_to_yolo(valid_json, frames_dir, output_dir, "val")
    
    # Handle test split (no labels)
    logger.info("Processing test split (no labels)...")
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    test_images_dir = output_dir / "images" / "test"
    test_labels_dir = output_dir / "labels" / "test"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_info in tqdm(test_data['images'], desc="Processing test"):
        filename = img_info['file_name']
        source_img = frames_dir / filename
        if not source_img.exists():
            source_img = frames_dir / "frames" / filename
        if source_img.exists():
            dest_img = test_images_dir / Path(filename).name
            if not dest_img.exists():
                shutil.copy2(source_img, dest_img)
    
    # Step 4: Create dataset.yaml
    logger.info("\n=== Step 4/4: Creating dataset configuration ===")
    create_dataset_yaml(output_dir, categories)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("✓ LTDv2 Dataset Download Complete!")
    logger.info("="*60)
    logger.info(f"Dataset location: {output_dir.absolute()}")
    logger.info(f"Train images: {len(list((output_dir / 'images' / 'train').glob('*')))}")
    logger.info(f"Val images: {len(list((output_dir / 'images' / 'val').glob('*')))}")
    logger.info(f"Test images: {len(list((output_dir / 'images' / 'test').glob('*')))}")
    logger.info(f"Classes: {len(categories)}")
    for cat in categories:
        logger.info(f"  - {cat['name']}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Verify: ls -lh {output_dir}/images/train | head")
    logger.info(f"  2. Train: ./venv/bin/python src/training/train.py --config configs/baseline.yaml")
    logger.info("="*60)


def download_ltdv2_subset(output_dir: Path, max_train: int = 10000, max_val: int = 1000):
    """Download a subset of LTDv2 for quick testing."""
    logger.info(f"Downloading LTDv2 subset (train={max_train}, val={max_val})...")
    logger.info("This will download annotations and extract a subset of images")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download annotation files only (much faster)
    logger.info("\n=== Step 1/3: Downloading annotations ===")
    train_json = download_file("data/Train.json", output_dir)
    valid_json = download_file("data/Valid.json", output_dir)
    frames_zip = download_file("data/frames.zip", output_dir)
    
    # Extract frames
    logger.info("\n=== Step 2/3: Extracting frames ===")
    frames_dir = extract_frames(frames_zip, output_dir)
    
    # Load annotations
    logger.info("\n=== Step 3/3: Creating subset ===")
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    # Take subset of images
    train_data['images'] = train_data['images'][:max_train]
    train_image_ids = {img['id'] for img in train_data['images']}
    train_data['annotations'] = [ann for ann in train_data['annotations'] 
                                   if ann['image_id'] in train_image_ids]
    
    # Save subset
    subset_train_json = output_dir / "Train_subset.json"
    with open(subset_train_json, 'w') as f:
        json.dump(train_data, f)
    
    # Convert to YOLO
    convert_coco_to_yolo(subset_train_json, frames_dir, output_dir, "train")
    
    # Validation subset
    with open(valid_json, 'r') as f:
        valid_data = json.load(f)
    
    valid_data['images'] = valid_data['images'][:max_val]
    valid_image_ids = {img['id'] for img in valid_data['images']}
    valid_data['annotations'] = [ann for ann in valid_data['annotations']
                                   if ann['image_id'] in valid_image_ids]
    
    subset_valid_json = output_dir / "Valid_subset.json"
    with open(subset_valid_json, 'w') as f:
        json.dump(valid_data, f)
    
    convert_coco_to_yolo(subset_valid_json, frames_dir, output_dir, "val")
    
    # Create empty test directory
    (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir, train_data['categories'])
    
    logger.info(f"\n✓ Subset created: {max_train} train, {max_val} val images")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download LTDv2 thermal object detection dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full dataset (~49 GB, 1M+ images)
  python download_ltdv2.py --output data/ltdv2 --mode full
  
  # Download subset for testing (~4 GB, 10K images)
  python download_ltdv2.py --output data/ltdv2_subset --mode subset --max-train 10000
  
  # Quick test (~400 MB, 1K images)
  python download_ltdv2.py --output data/ltdv2_test --mode subset --max-train 1000 --max-val 100
        """
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "subset"],
        default="full",
        help="Download mode: 'full' for complete dataset, 'subset' for smaller test dataset"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=10000,
        help="Maximum training images for subset mode (default: 10000)"
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=1000,
        help="Maximum validation images for subset mode (default: 1000)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    try:
        if args.mode == "full":
            download_ltdv2_full(output_dir)
        else:  # subset
            download_ltdv2_subset(output_dir, args.max_train, args.max_val)
        
        logger.info("\n✓ Download and preparation complete!")
        
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
