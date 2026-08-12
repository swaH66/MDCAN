"""Command-line unittest runner for MDCAN.

Examples:
    python test.py --device cpu --num-classes 3
    python test.py --data-path D:/datasets/ISIC2019 --num-classes 8 --device cuda
    python test.py --data-path D:/datasets/ISIC2019 --checkpoint best_model_f1.pth \
        --num-classes 8 --device cuda
"""

from __future__ import annotations

import argparse
import os
import unittest
from pathlib import Path

import torch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MDCAN unit tests")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Optional dataset root. It may contain test/<class>/<image>, or it "
            "may directly be a split directory containing <class>/<image>."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Examples: cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum number of real images used by the dataset test (default: 8)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2.")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.environ["MDCAN_TEST_DEVICE"] = args.device
    os.environ["MDCAN_TEST_NUM_CLASSES"] = str(args.num_classes)
    os.environ["MDCAN_TEST_MAX_SAMPLES"] = str(args.max_samples)
    if args.data_path is not None:
        os.environ["MDCAN_TEST_DATA_PATH"] = str(args.data_path.resolve())
    if args.checkpoint is not None:
        os.environ["MDCAN_TEST_CHECKPOINT"] = str(args.checkpoint.resolve())

    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
