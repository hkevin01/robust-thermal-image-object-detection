"""
YOLOv8 detector wrapper for thermal object detection.

This module provides a wrapper around Ultralytics YOLOv8 with custom
configurations for thermal imagery and temporal consistency.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ThermalYOLOv8:
    """
    YOLOv8 detector wrapper optimized for thermal imagery.

    Args:
        model_name: YOLOv8 variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
        num_classes: Number of detection classes
        pretrained: Whether to use pretrained weights
        device: Device to run model on ('cuda' or 'cpu')
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
    """

    def __init__(
        self,
        model_name: str = "yolov8m",
        num_classes: int = 4,
        pretrained: bool = True,
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing {model_name} for {num_classes} classes on {self.device}"
        )

        # Load model
        try:
            if pretrained:
                self.model = YOLO(f"{model_name}.pt")
                logger.info(f"Loaded pretrained {model_name}")
            else:
                self.model = YOLO(f"{model_name}.yaml")
                logger.info(f"Initialized {model_name} from scratch")

            # Move to device
            self.model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Failed to initialize YOLOv8: {e}") from e

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        save_dir: str = "runs/train",
        resume: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data_yaml: Path to data YAML configuration
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            save_dir: Directory to save results
            resume: Resume from last checkpoint
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")

        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=save_dir,
                name="exp",
                resume=resume,
                device=self.device,
                verbose=True,
                **kwargs,
            )

            logger.info("Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e

    def predict(
        self,
        images: Any,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on images.

        Args:
            images: Image(s) to run inference on (path, array, tensor, or list)
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold for NMS (uses default if None)
            augment: Use test-time augmentation
            **kwargs: Additional prediction arguments

        Returns:
            List of prediction dictionaries with 'boxes', 'scores', 'labels'
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        try:
            results = self.model.predict(
                images,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
                augment=augment,
                **kwargs,
            )

            # Convert to standard format
            predictions = []
            for result in results:
                boxes = result.boxes

                pred_dict = {
                    "boxes": boxes.xyxy.cpu(),  # [x1, y1, x2, y2]
                    "scores": boxes.conf.cpu(),
                    "labels": boxes.cls.cpu().long(),
                }

                predictions.append(pred_dict)

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def validate(
        self, data_yaml: str, batch_size: int = 16, img_size: int = 640, **kwargs
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            data_yaml: Path to data YAML configuration
            batch_size: Batch size for validation
            img_size: Input image size
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics dictionary
        """
        logger.info("Running validation")

        try:
            results = self.model.val(
                data=data_yaml,
                batch=batch_size,
                imgsz=img_size,
                device=self.device,
                verbose=True,
                **kwargs,
            )

            # Extract key metrics
            metrics = {
                "map50": results.box.map50,
                "map50-95": results.box.map,
                "precision": results.box.mp,
                "recall": results.box.mr,
            }

            logger.info(f"Validation mAP@0.5: {metrics['map50']:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise RuntimeError(f"Validation failed: {e}") from e

    def save(self, save_path: str) -> None:
        """
        Save model weights.

        Args:
            save_path: Path to save model
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            self.model.save(str(save_path))
            logger.info(f"Model saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load(self, checkpoint_path: str) -> None:
        """
        Load model weights.

        Args:
            checkpoint_path: Path to checkpoint
        """
        try:
            self.model = YOLO(checkpoint_path)
            self.model.to(self.device)
            logger.info(f"Model loaded from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def export(
        self, format: str = "onnx", save_path: Optional[str] = None, **kwargs
    ) -> str:
        """
        Export model to different format.

        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            save_path: Optional custom save path
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        logger.info(f"Exporting model to {format} format")

        try:
            export_path = self.model.export(format=format, **kwargs)
            logger.info(f"Model exported to {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        try:
            info = {
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "device": self.device,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "num_trainable": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
            }
            return info

        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {}


def create_data_yaml(
    train_path: str,
    val_path: str,
    test_path: Optional[str],
    class_names: List[str],
    output_path: str,
) -> str:
    """
    Create YOLO data configuration YAML file.

    Args:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Optional path to test images
        class_names: List of class names
        output_path: Path to save YAML file

    Returns:
        Path to created YAML file
    """
    import yaml

    data_config = {
        "path": str(Path(train_path).parent.parent),
        "train": str(Path(train_path).name),
        "val": str(Path(val_path).name),
        "nc": len(class_names),
        "names": class_names,
    }

    if test_path:
        data_config["test"] = str(Path(test_path).name)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    logger.info(f"Created data YAML at {output_path}")
    return str(output_path)
