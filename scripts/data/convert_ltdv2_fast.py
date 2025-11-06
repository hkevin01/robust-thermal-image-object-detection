#!/usr/bin/env python3
"""
Fast LTDv2 COCO to YOLO converter using symlinks.
"""
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def convert_coco_to_yolo_fast(coco_json_path: Path, frames_dir: Path, output_dir: Path, split: str):
    """Convert COCO JSON to YOLO format using symlinks."""
    logger.info(f"Converting {split} annotations to YOLO format...")
    
    # Load COCO annotations
    logger.info(f"Loading {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Build mappings
    image_map = {img['id']: img for img in coco_data['images']}
    category_map = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    logger.info(f"Processing {len(image_map)} images...")
    errors = 0
    processed = 0
    
    for img_id, img_info in tqdm(image_map.items(), desc=f"Converting {split}"):
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Find source image - check nested path first
        source_img = frames_dir / "frames" / filename.replace("frames/", "")
        if not source_img.exists():
            source_img = frames_dir / filename
        if not source_img.exists():
            errors += 1
            if errors < 10:  # Only show first 10 errors
                logger.warning(f"Image not found: {filename}")
            continue
        
        # Create symlink to image (much faster than copy)
        dest_img = images_dir / Path(filename).name
        if not dest_img.exists():
            try:
                dest_img.symlink_to(source_img.absolute())
            except Exception as e:
                if errors < 10:
                    logger.warning(f"Failed to symlink {filename}: {e}")
                errors += 1
                continue
        
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
            # Create empty label file
            label_file.touch()
        
        processed += 1
    
    logger.info(f"✓ Converted {split}: {processed} images, {errors} errors")
    return processed, errors

def main():
    # Paths
    base_dir = Path("/home/kevin/Projects/robust-thermal-image-object-detection/data/ltdv2_full")
    cache_dir = base_dir / ".cache/datasets--vapaau--LTDv2"
    
    # Find JSON files by hash
    blobs_dir = cache_dir / "blobs"
    train_json = blobs_dir / "9a197d91cb2f86a4ddede6abaab1ceeeb07b1b53fa3e649350fd283e7d403096"  # Train.json (961M)
    valid_json = blobs_dir / "69dcb2841a429819ecec642848fb9c8cbccdb6314177bf2411fa5d0152acb9fa"  # Valid.json (128M)
    test_json = blobs_dir / "c64c231d5a59cc4cba832e009c2d53eca24b53de22bf9d486abb24edf85ff8f3"   # TestNoLabels.json (32M)
    
    frames_dir = base_dir / "frames"
    
    logger.info("="*60)
    logger.info("Starting Fast LTDv2 Conversion (using symlinks)")
    logger.info("="*60)
    
    # Load categories from train JSON
    logger.info("Loading category information...")
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    categories = train_data['categories']
    
    # Convert splits
    train_processed, train_errors = convert_coco_to_yolo_fast(train_json, frames_dir, base_dir, "train")
    val_processed, val_errors = convert_coco_to_yolo_fast(valid_json, frames_dir, base_dir, "val")
    
    # Handle test split (no labels)
    logger.info("Processing test split (no labels)...")
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    test_images_dir = base_dir / "images" / "test"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    test_processed = 0
    test_errors = 0
    for img_info in tqdm(test_data['images'], desc="Processing test"):
        filename = img_info['file_name']
        source_img = frames_dir / "frames" / filename.replace("frames/", "")
        if not source_img.exists():
            source_img = frames_dir / filename
        if source_img.exists():
            dest_img = test_images_dir / Path(filename).name
            if not dest_img.exists():
                try:
                    dest_img.symlink_to(source_img.absolute())
                    test_processed += 1
                except Exception as e:
                    test_errors += 1
        else:
            test_errors += 1
    
    # Create dataset.yaml
    logger.info("Creating dataset configuration...")
    yaml_content = f"""# LTDv2 Dataset Configuration
# Thermal object detection dataset with 1M+ images
# Source: https://huggingface.co/datasets/vapaau/LTDv2

path: {base_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(categories)}
names:
"""
    for idx, cat in enumerate(categories):
        yaml_content += f"  {idx}: {cat['name']}\n"
    
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("✓ LTDv2 Conversion Complete!")
    logger.info("="*60)
    logger.info(f"Train: {train_processed} images ({train_errors} errors)")
    logger.info(f"Val: {val_processed} images ({val_errors} errors)")
    logger.info(f"Test: {test_processed} images ({test_errors} errors)")
    logger.info(f"Classes: {len(categories)}")
    for cat in categories:
        logger.info(f"  - {cat['name']}")
    logger.info(f"\nDataset: {base_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
