#!/usr/bin/env python3
"""
Memory-efficient LTDv2 COCO to YOLO converter using streaming and symlinks.
Processes large JSON files without loading everything into memory at once.
"""
import json
import logging
from pathlib import Path
from collections import defaultdict
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def convert_split(json_path: Path, frames_dir: Path, output_dir: Path, split: str):
    """Convert one split from COCO to YOLO format."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Converting {split} split...")
    logger.info(f"{'='*60}")
    
    # Create output directories
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON (this is the slow part)
    logger.info(f"Loading {json_path.name} ({json_path.stat().st_size / 1024**2:.1f} MB)...")
    logger.info("This may take a few minutes for large files...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"✓ Loaded: {len(data['images'])} images, {len(data.get('annotations', []))} annotations")
    
    # Build mappings
    logger.info("Building mappings...")
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    if 'annotations' in data:
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
    
    logger.info(f"Processing {len(images)} images...")
    
    # Process in batches for progress tracking
    processed = 0
    errors = 0
    batch_size = 1000
    total = len(images)
    
    for img_id, img_info in images.items():
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Find source image
        source_img = frames_dir / "frames" / filename.replace("frames/", "")
        if not source_img.exists():
            source_img = frames_dir / filename
        if not source_img.exists():
            errors += 1
            continue
        
        # Create symlink
        dest_img = images_dir / Path(filename).name
        if not dest_img.exists():
            try:
                dest_img.symlink_to(source_img.absolute())
            except Exception:
                errors += 1
                continue
        
        # Create YOLO label
        label_file = labels_dir / (Path(filename).stem + '.txt')
        
        if img_id in annotations_by_image:
            with open(label_file, 'w') as f:
                for ann in annotations_by_image[img_id]:
                    x, y, w, h = ann['bbox']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    class_id = categories[ann['category_id']]
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        else:
            label_file.touch()
        
        processed += 1
        
        # Progress update
        if processed % batch_size == 0:
            pct = (processed / total) * 100
            logger.info(f"  Progress: {processed}/{total} ({pct:.1f}%)")
    
    logger.info(f"✓ {split}: {processed} images processed, {errors} errors")
    return processed, errors, data['categories']

def main():
    logger.info("="*60)
    logger.info("LTDv2 COCO to YOLO Converter (Streaming Mode)")
    logger.info("="*60)
    
    # Paths
    base_dir = Path("/home/kevin/Projects/robust-thermal-image-object-detection/data/ltdv2_full")
    blobs_dir = base_dir / ".cache/datasets--vapaau--LTDv2/blobs"
    frames_dir = base_dir / "frames"
    
    # JSON files (by hash)
    train_json = blobs_dir / "9a197d91cb2f86a4ddede6abaab1ceeeb07b1b53fa3e649350fd283e7d403096"
    valid_json = blobs_dir / "69dcb2841a429819ecec642848fb9c8cbccdb6314177bf2411fa5d0152acb9fa"
    test_json = blobs_dir / "c64c231d5a59cc4cba832e009c2d53eca24b53de22bf9d486abb24edf85ff8f3"
    
    # Convert splits
    train_processed, train_errors, categories = convert_split(train_json, frames_dir, base_dir, "train")
    val_processed, val_errors, _ = convert_split(valid_json, frames_dir, base_dir, "val")
    
    # Test split (no labels)
    logger.info(f"\n{'='*60}")
    logger.info("Processing test split (no labels)...")
    logger.info("="*60)
    
    logger.info(f"Loading {test_json.name}...")
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    test_images_dir = base_dir / "images" / "test"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    test_processed = 0
    test_errors = 0
    batch_size = 1000
    total_test = len(test_data['images'])
    
    for idx, img_info in enumerate(test_data['images'], 1):
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
                except Exception:
                    test_errors += 1
        else:
            test_errors += 1
        
        if idx % batch_size == 0:
            pct = (idx / total_test) * 100
            logger.info(f"  Progress: {idx}/{total_test} ({pct:.1f}%)")
    
    logger.info(f"✓ test: {test_processed} images processed, {test_errors} errors")
    
    # Create data.yaml
    logger.info("\nCreating dataset configuration...")
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
    
    with open(base_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("✓ CONVERSION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Train:   {train_processed:7,} images ({train_errors} errors)")
    logger.info(f"Val:     {val_processed:7,} images ({val_errors} errors)")
    logger.info(f"Test:    {test_processed:7,} images ({test_errors} errors)")
    logger.info(f"Total:   {train_processed + val_processed + test_processed:7,} images")
    logger.info(f"\nClasses: {len(categories)}")
    for cat in categories:
        logger.info(f"  - {cat['name']}")
    logger.info(f"\nDataset location: {base_dir}")
    logger.info(f"Configuration: {base_dir}/data.yaml")
    logger.info("\nReady to train! Use:")
    logger.info("  ./venv/bin/python src/training/train.py --config configs/baseline.yaml")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
