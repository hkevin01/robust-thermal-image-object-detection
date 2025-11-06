"""
Evaluation runner stub.

Compute metrics using TemporalDetectionMetrics and write a report.
"""

import argparse
import logging
from pathlib import Path

from src.evaluation.metrics import TemporalDetectionMetrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object detector")
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--data", type=str, required=False)
    parser.add_argument("--output", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation (stub)")

    metrics = TemporalDetectionMetrics()

    # This is where predictions and targets would be loaded and fed
    # For now, compute empty report
    report = metrics.compute_all_metrics()

    report_file = out_dir / "metrics_report.json"
    import json

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved evaluation report to {report_file}")


if __name__ == "__main__":
    main()
