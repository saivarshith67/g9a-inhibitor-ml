#!/usr/bin/env python
"""Step 04 — train 5 classifiers, print/save holdout + CV metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.models.train import run_compare_and_save
from g9a_ml.paths import dataset_stem, ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    stem = dataset_stem(cfg)
    path = cfg["processed_dir"] / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/03_balance.py first.")

    df = pd.read_csv(path)
    print(f"Loaded {path} shape={df.shape}")
    holdout, cv = run_compare_and_save(
        df,
        metrics_dir=cfg["metrics_dir"],
        stem=stem,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
        cv_folds=cfg["train"]["cv_folds"],
    )
    print("\n=== Holdout metrics (sorted by accuracy) ===")
    print(holdout.to_string(index=False))
    print("\n=== Cross-validation accuracy ===")
    print(cv.to_string(index=False))
    print(f"\nSaved under {cfg['metrics_dir']}")


if __name__ == "__main__":
    main()
