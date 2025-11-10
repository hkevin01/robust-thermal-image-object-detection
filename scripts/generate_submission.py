#!/usr/bin/env python3
"""
Generate Competition Submission for WACV 2026 RWS Challenge
============================================================

Generates COCO JSON format predictions for submission to:
https://www.codabench.org/competitions/10954/

Output format:
[
    {
        "image_id": int,
        "category_id": int (1-4),
        "bbox": [x, y, width, height],  # Absolute pixels
        "score": float (0.0-1.0)
    },
    ...
]
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import torch

# Add patches if needed
sys.path.insert(0, str(Path(__file__).parent.parent / 'patches'))

try:
    from conv2d_fallback import patch_torch_conv2d
    patch_torch_conv2d()
    print("✅ Conv2d patch applied")
except:
    print("ℹ️  No Conv2d patch needed")

from ultralytics import YOLO


# Class mapping: YOLO → LTDv2
# LTDv2 uses 4 classes: Person, Bicycle, Motorcycle, Vehicle
YOLO_TO_LTD = {
    0: 1,  # person → Person
    1: 2,  # bicycle → Bicycle
    3: 3,  # motorcycle → Motorcycle
    2: 4,  # car → Vehicle
    5: 4,  # bus → Vehicle
    7: 4,  # truck → Vehicle
}


def generate_predictions(
    model_path: str,
    test_images_dir: str,
    output_file: str,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    device: str = '0',
    batch_size: int = 4,
    img_size: int = 640
):
    """
    Generate predictions for all test images.
    
    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        test_images_dir: Directory containing test images
        output_file: Output JSON file path
        conf_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
        device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
        batch_size: Batch size for inference
        img_size: Image size for inference
    """
    
    print("="*70)
    print("WACV 2026 RWS Submission Generator")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Test Images: {test_images_dir}")
    print(f"Output: {output_file}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load model
    print("\n📦 Loading model...")
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Get test images
    test_dir = Path(test_images_dir)
    image_files = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {test_images_dir}")
        sys.exit(1)
    
    print(f"✅ Found {len(image_files)} test images")
    
    # Run inference
    print("\n🔮 Running inference...")
    predictions = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Extract image ID from filename
        # Assuming format: {image_id}.jpg or similar
        image_id = int(img_file.stem)
        
        # Run prediction
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            device=device,
            verbose=False
        )
        
        # Extract detections
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Process each detection
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to COCO format (x, y, width, height)
                x = float(x1)
                y = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                # Get class and confidence
                yolo_class = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Map YOLO class to LTDv2 class
                if yolo_class not in YOLO_TO_LTD:
                    continue  # Skip classes not in LTDv2
                
                ltd_class = YOLO_TO_LTD[yolo_class]
                
                # Create prediction entry
                pred = {
                    "image_id": image_id,
                    "category_id": ltd_class,
                    "bbox": [x, y, width, height],
                    "score": confidence
                }
                
                predictions.append(pred)
    
    print(f"✅ Generated {len(predictions)} predictions across {len(image_files)} images")
    print(f"   Average: {len(predictions) / len(image_files):.1f} detections per image")
    
    # Save predictions
    print(f"\n💾 Saving predictions to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Predictions saved: {output_file}")
    
    # Print statistics
    print("\n📊 Prediction Statistics:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Total images: {len(image_files)}")
    
    # Count per class
    class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for pred in predictions:
        class_counts[pred['category_id']] += 1
    
    CLASS_NAMES = {1: "Person", 2: "Bicycle", 3: "Motorcycle", 4: "Vehicle"}
    print("\n   Per-class predictions:")
    for class_id, count in sorted(class_counts.items()):
        print(f"     {CLASS_NAMES[class_id]}: {count}")
    
    # Confidence distribution
    scores = [p['score'] for p in predictions]
    if scores:
        print(f"\n   Confidence scores:")
        print(f"     Min: {min(scores):.3f}")
        print(f"     Max: {max(scores):.3f}")
        print(f"     Mean: {sum(scores)/len(scores):.3f}")
    
    print("\n" + "="*70)
    print("✅ Submission generation complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Validate submission: python scripts/validate_submission.py {output_file}")
    print(f"2. Upload to: https://www.codabench.org/competitions/10954/")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate competition submission")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--test-dir", required=True, help="Directory with test images")
    parser.add_argument("--output", default="submission.json", help="Output JSON file")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IOU threshold")
    parser.add_argument("--device", default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    generate_predictions(
        model_path=args.model,
        test_images_dir=args.test_dir,
        output_file=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        batch_size=args.batch,
        img_size=args.img_size
    )
