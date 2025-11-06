#!/usr/bin/env python3
"""
Fixed LTDv2 COCO to YOLO converter - preserves directory structure to avoid filename collisions.
"""
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def load_json(json_path: Path):
    """Load JSON file."""
    logger.info(f"Loading {json_path.name} ({json_path.stat().st_size / 1024**2:.0f} MB)...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"✓ Loaded: {len(data['images']):,} images")
    return data

def convert_split(json_path: Path, frames_dir: Path, output_dir: Path, split: str):
    """Convert one split from COCO to YOLO format."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Converting {split.upper()} split")
    logger.info(f"{'='*60}")
    
    # Create output directories
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON
    data = load_json(json_path)
    
    if 'annotations' in data:
        logger.info(f"Annotations: {len(data['annotations']):,}")
    
    # Build category mapping
    categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
    
    # Group annotations by image ID
    logger.info("Grouping annotations...")
    annotations_by_image = {}
    
    if 'annotations' in data:
        chunk_size = 100000
        total_ann = len(data['annotations'])
        
        for i in range(0, total_ann, chunk_size):
            chunk = data['annotations'][i:i+chunk_size]
            for ann in chunk:
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
            
            if (i + chunk_size) % (chunk_size * 5) == 0:
                logger.info(f"  {min(i+chunk_size, total_ann):,}/{total_ann:,} annotations")
        
        logger.info(f"✓ Grouped {total_ann:,} annotations")
    
    # Process images - PRESERVE DIRECTORY STRUCTURE
    logger.info(f"Processing {len(data['images']):,} images...")
    
    processed = 0
    errors = 0
    batch_size = 5000
    
    for idx, img_info in enumerate(data['images'], 1):
        img_id = img_info['id']
        # Original: frames/20200514/clip_22_2307/image_0015.jpg
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Remove 'frames/' prefix if present, keep rest of path
        relative_path = filename.replace("frames/", "")  # e.g., 20200514/clip_22_2307/image_0015.jpg
        
        # Find source image
        source_img = frames_dir / "frames" / relative_path
        if not source_img.exists():
            source_img = frames_dir / filename
        if not source_img.exists():
            errors += 1
            continue
        
        # Create subdirectories in output to match structure
        # Convert path separators to underscores for flat structure
        # 20200514/clip_22_2307/image_0015.jpg -> 20200514_clip_22_2307_image_0015.jpg
        flat_filename = relative_path.replace("/", "_")
        
        dest_img = images_dir / flat_filename
        dest_label = labels_dir / (Path(flat_filename).stem + '.txt')
        
        # Create symlink
        if not dest_img.exists():
            try:
                dest_img.symlink_to(source_img.absolute())
            except Exception as e:
                errors += 1
                continue
        
        # Create YOLO label
        if img_id in annotations_by_image:
            with open(dest_label, 'w') as f:
                for ann in annotations_by_image[img_id]:
                    x, y, w, h = ann['bbox']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    class_id = categories[ann['category_id']]
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        else:
            dest_label.touch()
        
        processed += 1
        
        # Progress update
        if idx % batch_size == 0:
            pct = (idx / len(data['images'])) * 100
            logger.info(f"  {idx:,}/{len(data['images']):,} ({pct:.1f}%)")
    
    logger.info(f"✓ {split.upper()}: {processed:,} images, {errors:,} errors")
    return processed, errors, data['categories']

def main():
    logger.info("="*60)
    logger.info("LTDv2 COCO → YOLO Converter (FIXED)")
    logger.info("="*60)
    
    # Paths
    base_dir = Path("/home/kevin/Projects/robust-thermal-image-object-detection/data/ltdv2_full")
    blobs_dir = base_dir / ".cache/datasets--vapaau--LTDv2/blobs"
    frames_dir = base_dir / "frames"
    
    # JSON files
    train_json = blobs_dir / "9a197d91cb2f86a4ddede6abaab1ceeeb07b1b53fa3e649350fd283e7d403096"
    valid_json = blobs_dir / "69dcb2841a429819ecec642848fb9c8cbccdb6314177bf2411fa5d0152acb9fa"
    test_json = blobs_dir / "c64c231d5a59cc4cba832e009c2d53eca24b53de22bf9d486abb24edf85ff8f3"
    
    # Convert splits
    train_processed, train_errors, categories = convert_split(train_json, frames_dir, base_dir, "train")
    val_processed, val_errors, _ = convert_split(valid_json, frames_dir, base_dir, "val")
    
    # Test split
    logger.info(f"\n{'='*60}")
    logger.info("Processing TEST split (no labels)")
    logger.info("="*60)
    
    test_data = load_json(test_json)
    
    test_images_dir = base_dir / "images" / "test"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    test_processed = 0
    test_errors = 0
    
    for idx, img_info in enumerate(test_data['images'], 1):
        filename = img_info['file_name']
        relative_path = filename.replace("frames/", "")
        
        source_img = frames_dir / "frames" / relative_path
        if not source_img.exists():
            source_img = frames_dir / filename
        
        if source_img.exists():
            flat_filename = relative_path.replace("/", "_")
            dest_img = test_images_dir / flat_filename
            
            if not dest_img.exists():
                try:
                    dest_img.symlink_to(source_img.absolute())
                    test_processed += 1
                except:
                    test_errors += 1
        else:
            test_errors += 1
        
        if idx % 5000 == 0:
            pct = (idx / len(test_data['images'])) * 100
            logger.info(f"  {idx:,}/{len(test_data['images']):,} ({pct:.1f}%)")
    
    logger.info(f"✓ TEST: {test_processed:,} images, {test_errors:,} errors")
    
    # Create data.yaml
    yaml_content = f"""# LTDv2 Dataset - YOLO Format
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
    
    # Summary
    total = train_processed + val_processed + test_processed
    logger.info("\n" + "="*60)
    logger.info("✓ CONVERSION COMPLETE")
    logger.info("="*60)
    logger.info(f"Train:  {train_processed:8,} images ({train_errors:,} errors)")
    logger.info(f"Val:    {val_processed:8,} images ({val_errors:,} errors)")
    logger.info(f"Test:   {test_processed:8,} images ({test_errors:,} errors)")
    logger.info(f"TOTAL:  {total:8,} images")
    logger.info(f"\nDataset: {base_dir}")
    logger.info(f"Config:  {base_dir}/data.yaml")
    logger.info("\nReady for training!")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
