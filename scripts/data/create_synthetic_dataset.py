"""
Create a synthetic thermal dataset for testing and development.

This script generates a small dataset with realistic structure for:
- Testing the training pipeline
- Validating configurations
- Quick experiments before downloading full LTDv2 dataset
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def generate_thermal_image(width=640, height=480):
    """Generate a synthetic thermal image."""
    # Create grayscale thermal image with some structure
    img = np.random.randint(100, 200, (height, width), dtype=np.uint8)
    
    # Add some thermal "hotspots" (simulating people/vehicles)
    num_hotspots = random.randint(1, 4)
    for _ in range(num_hotspots):
        x = random.randint(50, width - 100)
        y = random.randint(50, height - 100)
        w = random.randint(40, 120)
        h = random.randint(60, 150)
        
        # Create hotspot with gradient
        for i in range(h):
            for j in range(w):
                if y + i < height and x + j < width:
                    distance = np.sqrt((i - h/2)**2 + (j - w/2)**2)
                    max_dist = np.sqrt((h/2)**2 + (w/2)**2)
                    intensity = int(255 - (distance / max_dist) * 100)
                    img[y + i, x + j] = min(255, max(img[y + i, x + j], intensity))
    
    # Convert to RGB (thermal cameras often display as grayscale)
    img_rgb = np.stack([img, img, img], axis=-1)
    return Image.fromarray(img_rgb)


def generate_yolo_annotations(image_id, width=640, height=480, num_objects=None):
    """Generate YOLO format annotations."""
    if num_objects is None:
        num_objects = random.randint(1, 5)
    
    annotations = []
    classes = [0, 1, 2, 3]  # Person, Bicycle, Motorcycle, Vehicle
    
    for _ in range(num_objects):
        cls = random.choice(classes)
        
        # Generate bounding box (normalized coordinates)
        x_center = random.uniform(0.1, 0.9)
        y_center = random.uniform(0.1, 0.9)
        bbox_width = random.uniform(0.05, 0.3)
        bbox_height = random.uniform(0.08, 0.4)
        
        # Ensure bbox stays within image
        x_center = max(bbox_width/2, min(1 - bbox_width/2, x_center))
        y_center = max(bbox_height/2, min(1 - bbox_height/2, y_center))
        
        annotations.append(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    return annotations


def create_synthetic_dataset(
    output_dir: str,
    num_train: int = 100,
    num_val: int = 30,
    num_test: int = 20,
    image_size: tuple = (640, 480)
):
    """Create a complete synthetic dataset."""
    output_path = Path(output_dir)
    
    print(f"Creating synthetic dataset at: {output_path}")
    print(f"  - Train: {num_train} images")
    print(f"  - Val: {num_val} images")
    print(f"  - Test: {num_test} images")
    
    # Create directory structure
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    
    for split, num_images in splits.items():
        # Create directories
        img_dir = output_path / 'images' / split
        label_dir = output_path / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {split} split...")
        for i in range(num_images):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_images}")
            
            # Generate image
            img_name = f"thermal_{split}_{i:04d}.jpg"
            img = generate_thermal_image(image_size[0], image_size[1])
            img.save(img_dir / img_name)
            
            # Generate annotations
            label_name = f"thermal_{split}_{i:04d}.txt"
            annotations = generate_yolo_annotations(
                image_id=img_name,
                width=image_size[0],
                height=image_size[1]
            )
            
            with open(label_dir / label_name, 'w') as f:
                f.write('\n'.join(annotations))
    
    # Create data.yaml
    data_yaml_content = f"""# Synthetic Thermal Dataset Configuration
# Generated for testing and development

path: {output_path.absolute()}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')
test: images/test    # Test images (relative to 'path')

# Number of classes
nc: 4

# Class names
names:
  0: Person
  1: Bicycle
  2: Motorcycle
  3: Vehicle
"""
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✓ Dataset created successfully!")
    print(f"✓ Data config saved to: {output_path / 'data.yaml'}")
    print(f"\nDataset statistics:")
    print(f"  Total images: {num_train + num_val + num_test}")
    print(f"  Train: {num_train} ({num_train/(num_train+num_val+num_test)*100:.1f}%)")
    print(f"  Val: {num_val} ({num_val/(num_train+num_val+num_test)*100:.1f}%)")
    print(f"  Test: {num_test} ({num_test/(num_train+num_val+num_test)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create synthetic thermal dataset")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='Output directory for synthetic dataset'
    )
    parser.add_argument(
        '--num-train',
        type=int,
        default=100,
        help='Number of training images'
    )
    parser.add_argument(
        '--num-val',
        type=int,
        default=30,
        help='Number of validation images'
    )
    parser.add_argument(
        '--num-test',
        type=int,
        default=20,
        help='Number of test images'
    )
    parser.add_argument(
        '--small',
        action='store_true',
        help='Create small dataset (20 train, 5 val, 5 test) for quick testing'
    )
    
    args = parser.parse_args()
    
    if args.small:
        print("Creating SMALL dataset for quick testing...")
        create_synthetic_dataset(
            output_dir=args.output_dir,
            num_train=20,
            num_val=5,
            num_test=5
        )
    else:
        create_synthetic_dataset(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test
        )
