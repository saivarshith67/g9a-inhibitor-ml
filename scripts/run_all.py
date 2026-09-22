#!/usr/bin/env python
"""Run the full pipeline (steps 01–06) for the configured dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "01_download_coordinates.py",
    "02_build_features.py",
    "03_balance.py",
    "04_compare_classifiers.py",
    "05_feature_importance.py",
    "06_train_final.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip coordinate download (files must already exist).",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Start from step N (1–6).",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=6,
        help="Stop after step N (1–6).",
    )
    parser.add_argument(
        "--sweep-depth",
        action="store_true",
        help="Pass --sweep-depth to step 06.",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Limit IDs for step 01 smoke-test download.",
    )
    args = parser.parse_args()

    for i, name in enumerate(SCRIPTS, start=1):
        if i < args.from_step or i > args.to_step:
            continue
        if args.skip_download and i == 1:
            print(f"[skip] step {i}: {name}")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / name)]
        if args.config:
            cmd += ["--config", args.config]
        if i == 1 and args.download_limit is not None:
            cmd += ["--limit", str(args.download_limit)]
        if i == 6 and args.sweep_depth:
            cmd.append("--sweep-depth")
        print(f"\n===== Running step {i}: {name} =====")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
