#!/usr/bin/env python3
"""
Training entry point for thermal object detection.

This script orchestrates the complete training pipeline using the
ThermalDetectorTrainer class with W&B integration.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.trainer import ThermalDetectorTrainer, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train thermal object detector with YOLOv8"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="thermal-detection",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/team name"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B tracking"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (no training)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path for evaluation"
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Initialize trainer
    try:
        trainer = ThermalDetectorTrainer(
            config=config,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)

    # Run training or evaluation
    try:
        if args.eval_only:
            logger.info("Running evaluation only")
            data_yaml = config.get("training", {}).get("data_yaml")
            if not data_yaml:
                logger.error("data_yaml not specified in config")
                sys.exit(1)

            metrics = trainer.evaluate(
                data_yaml=data_yaml, checkpoint_path=args.checkpoint
            )

            logger.info("=" * 50)
            logger.info("Evaluation Results:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            logger.info("=" * 50)

        else:
            logger.info("Starting training")
            results = trainer.train()

            logger.info("=" * 50)
            logger.info("Training completed successfully!")
            logger.info(f"Training time: {results['training_time']:.2f}s")
            logger.info(f"Best model: {results['best_model_path']}")
            logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training/evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
