#!/usr/bin/env python3
"""
Download LTDv2 dataset from HuggingFace.

This script downloads the LTDv2 dataset with robust error handling,
resume capabilities, and progress tracking.

Dataset structure:
- frames.zip (48 GB): All thermal images
- Train.json (1 GB): COCO format training annotations
- Valid.json (134 MB): COCO format validation annotations
- TestNoLabels.json (33 MB): Test split without labels
"""

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed")
    print("Run: pip install huggingface-hub tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset_streaming(output_dir: Path, split: str = "train") -> None:
    """
    Download dataset using streaming (memory efficient).

    Args:
        output_dir: Output directory path
        split: Dataset split to download
    """
    logger.info(f"Downloading LTDv2 dataset (streaming mode) to {output_dir}")

    try:
        dataset = load_dataset(
            "vapaau/LTDv2", split=split, streaming=True, trust_remote_code=True
        )

        logger.info(f"Dataset loaded in streaming mode")
        logger.info("Use this dataset directly in your training pipeline")
        logger.info("Example: for batch in dataset: ...")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def download_dataset_full(
    output_dir: Path, split: Optional[str] = None, max_workers: int = 4
) -> None:
    """
    Download full dataset to disk.

    Args:
        output_dir: Output directory path
        split: Optional specific split to download
        max_workers: Number of parallel download workers
    """
    logger.info(f"Downloading LTDv2 dataset (full mode) to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download using snapshot_download for better control
        logger.info("Starting dataset download...")

        cache_dir = output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        downloaded_path = snapshot_download(
            repo_id="vapaau/LTDv2",
            repo_type="dataset",
            cache_dir=str(cache_dir),
            resume_download=True,
            max_workers=max_workers,
        )

        logger.info(f"Dataset downloaded to: {downloaded_path}")

        # Load dataset to verify
        if split:
            dataset = load_dataset("vapaau/LTDv2", split=split, cache_dir=str(cache_dir))
        else:
            dataset = load_dataset("vapaau/LTDv2", cache_dir=str(cache_dir))

        logger.info(f"Dataset loaded successfully")
        logger.info(f"Dataset info: {dataset}")

        # Save dataset statistics
        stats_file = output_dir / "dataset_stats.txt"
        with open(stats_file, "w") as f:
            f.write(f"LTDv2 Dataset Statistics\n")
            f.write(f"========================\n\n")
            f.write(f"Download path: {downloaded_path}\n")
            if isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    f.write(f"\n{split_name} split:\n")
                    f.write(f"  Samples: {len(split_data)}\n")
            else:
                f.write(f"Samples: {len(dataset)}\n")

        logger.info(f"Dataset statistics saved to {stats_file}")

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def download_specific_files(output_dir: Path, files: list) -> None:
    """
    Download specific files from the dataset.

    Args:
        output_dir: Output directory path
        files: List of file paths to download
    """
    logger.info(f"Downloading {len(files)} specific files...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(files, desc="Downloading files"):
        try:
            local_path = hf_hub_download(
                repo_id="vapaau/LTDv2",
                filename=file_path,
                repo_type="dataset",
                cache_dir=str(output_dir / "cache"),
                resume_download=True,
            )
            logger.info(f"Downloaded: {file_path} -> {local_path}")

        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download LTDv2 thermal object detection dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["streaming", "full", "files"],
        default="full",
        help="Download mode",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Specific split to download",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of download workers"
    )
    parser.add_argument(
        "--files", type=str, nargs="+", help="Specific files to download (files mode)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        if args.mode == "streaming":
            download_dataset_streaming(output_dir, args.split or "train")
        elif args.mode == "full":
            download_dataset_full(output_dir, args.split, args.max_workers)
        elif args.mode == "files":
            if not args.files:
                logger.error("--files argument required for files mode")
                sys.exit(1)
            download_specific_files(output_dir, args.files)

        logger.info("Download completed successfully!")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
