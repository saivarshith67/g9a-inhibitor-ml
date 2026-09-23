#!/usr/bin/env python
"""Run the full pipeline (steps 00–07) for the configured dataset.

Step list:
  00  check raw data files
  01  download missing coordinate JSONs
  02  engineer features → interim CSV
  03  balance classes → processed CSV
  04  exploratory data analysis
  05  compare classifiers
  06  feature importance
  07  train final RandomForest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "00_check_data.py",
    "01_download_coordinates.py",
    "02_build_features.py",
    "03_balance.py",
    "04_eda_dataset.py",
    "05_compare_classifiers.py",
    "06_feature_importance.py",
    "07_train_final.py",
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
        "--skip-check",
        action="store_true",
        help="Skip step 00 (data presence check).",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip step 04 (EDA).",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=0,
        help="Start from step N (0–7).",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=7,
        help="Stop after step N (0–7).",
    )
    parser.add_argument(
        "--sweep-depth",
        action="store_true",
        help="Pass --sweep-depth to step 07.",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Limit IDs for step 01 smoke-test download.",
    )
    args = parser.parse_args()

    for i, name in enumerate(SCRIPTS):
        if i < args.from_step or i > args.to_step:
            continue
        if args.skip_check and i == 0:
            print(f"[skip] step {i:02d}: {name}")
            continue
        if args.skip_download and i == 1:
            print(f"[skip] step {i:02d}: {name}")
            continue
        if args.skip_eda and i == 4:
            print(f"[skip] step {i:02d}: {name}")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / name)]
        if args.config:
            cmd += ["--config", args.config]
        if i == 1 and args.download_limit is not None:
            cmd += ["--limit", str(args.download_limit)]
        if i == 7 and args.sweep_depth:
            cmd.append("--sweep-depth")
        print(f"\n===== Running step {i:02d}: {name} =====")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
