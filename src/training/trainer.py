"""
Complete training pipeline with W&B integration and robust error handling.

This module orchestrates model training with experiment tracking, checkpointing,
early stopping, and comprehensive logging.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("W&B not installed. Install with: pip install wandb")

from src.evaluation.metrics import TemporalDetectionMetrics
from src.models.yolo_detector import ThermalYOLOv8

logger = logging.getLogger(__name__)


class ThermalDetectorTrainer:
    """
    Training orchestrator for thermal object detection.

    Handles training loop, validation, checkpointing, and experiment tracking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        use_wandb: bool = True,
        wandb_project: str = "thermal-detection",
        wandb_entity: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
            use_wandb: Whether to use Weights & Biases tracking
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
        """
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Initialize W&B if available
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config,
                    name=config.get("experiment_name", "baseline"),
                    tags=config.get("tags", []),
                )
                logger.info("Weights & Biases initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Initialize model
        self.model = self._build_model()

        # Initialize metrics
        self.metrics = TemporalDetectionMetrics(
            num_classes=config.get("num_classes", 4),
            iou_threshold=0.5,
            class_names=config.get("class_names", ["Person", "Bicycle", "Motorcycle", "Vehicle"]),
        )

        # Training state
        self.best_score = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0

        logger.info("Trainer initialized successfully")

    def _build_model(self) -> ThermalYOLOv8:
        """Build model from config."""
        model_config = self.config.get("model", {})

        model = ThermalYOLOv8(
            model_name=model_config.get("name", "yolov8m"),
            num_classes=self.config.get("num_classes", 4),
            pretrained=model_config.get("pretrained", True),
            device=model_config.get("device", None),
            conf_threshold=model_config.get("conf_threshold", 0.25),
            iou_threshold=model_config.get("iou_threshold", 0.45),
        )

        logger.info(f"Built model: {model.get_model_info()}")
        return model

    def train(self) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Returns:
            Training results dictionary
        """
        train_config = self.config.get("training", {})

        data_yaml = train_config.get("data_yaml")
        epochs = train_config.get("epochs", 100)
        batch_size = train_config.get("batch_size", 16)
        img_size = train_config.get("img_size", 640)
        save_dir = train_config.get("save_dir", "runs/train")

        # Additional training arguments
        kwargs = {
            "patience": train_config.get("patience", 50),
            "save_period": train_config.get("save_period", 10),
            "workers": train_config.get("workers", 8),
            "optimizer": train_config.get("optimizer", "auto"),
            "lr0": train_config.get("lr0", 0.01),
            "lrf": train_config.get("lrf", 0.01),
            "momentum": train_config.get("momentum", 0.937),
            "weight_decay": train_config.get("weight_decay", 0.0005),
            "warmup_epochs": train_config.get("warmup_epochs", 3.0),
            "warmup_momentum": train_config.get("warmup_momentum", 0.8),
            "warmup_bias_lr": train_config.get("warmup_bias_lr", 0.1),
            "box": train_config.get("box", 7.5),
            "cls": train_config.get("cls", 0.5),
            "dfl": train_config.get("dfl", 1.5),
            "pose": train_config.get("pose", 12.0),
            "kobj": train_config.get("kobj", 2.0),
            "label_smoothing": train_config.get("label_smoothing", 0.0),
            "nbs": train_config.get("nbs", 64),
            "hsv_h": train_config.get("hsv_h", 0.015),
            "hsv_s": train_config.get("hsv_s", 0.7),
            "hsv_v": train_config.get("hsv_v", 0.4),
            "degrees": train_config.get("degrees", 0.0),
            "translate": train_config.get("translate", 0.1),
            "scale": train_config.get("scale", 0.5),
            "shear": train_config.get("shear", 0.0),
            "perspective": train_config.get("perspective", 0.0),
            "flipud": train_config.get("flipud", 0.0),
            "fliplr": train_config.get("fliplr", 0.5),
            "mosaic": train_config.get("mosaic", 1.0),
            "mixup": train_config.get("mixup", 0.0),
            "copy_paste": train_config.get("copy_paste", 0.0),
            "amp": train_config.get("amp", True),
            "fraction": train_config.get("fraction", 1.0),
            "profile": train_config.get("profile", False),
            "overlap_mask": train_config.get("overlap_mask", True),
            "mask_ratio": train_config.get("mask_ratio", 4),
            "dropout": train_config.get("dropout", 0.0),
            "val": train_config.get("val", True),
        }

        logger.info("Starting training...")
        start_time = time.time()

        try:
            # Train model using YOLOv8's built-in training
            results = self.model.train(
                data_yaml=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                save_dir=save_dir,
                resume=train_config.get("resume", False),
                **kwargs,
            )

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")

            # Log final results to W&B
            if self.use_wandb:
                wandb.log(
                    {
                        "training_time_seconds": training_time,
                        "final_map50": results.results_dict.get("metrics/mAP50(B)", 0),
                        "final_map50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                    }
                )

            return {
                "training_time": training_time,
                "results": results,
                "best_model_path": str(Path(save_dir) / "exp" / "weights" / "best.pt"),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            if self.use_wandb:
                wandb.finish()

    def evaluate(
        self, data_yaml: str, checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation/test set.

        Args:
            data_yaml: Path to data configuration
            checkpoint_path: Optional path to model checkpoint

        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Starting evaluation...")

        try:
            # Load checkpoint if provided
            if checkpoint_path:
                self.model.load(checkpoint_path)

            # Run validation
            val_config = self.config.get("validation", {})
            metrics = self.model.validate(
                data_yaml=data_yaml,
                batch_size=val_config.get("batch_size", 16),
                img_size=val_config.get("img_size", 640),
            )

            logger.info(f"Evaluation mAP@0.5: {metrics['map50']:.4f}")

            # Log to W&B
            if self.use_wandb:
                wandb.log({"eval_map50": metrics["map50"], "eval_map50-95": metrics["map50-95"]})

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to config YAML

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved config to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise
