#!/usr/bin/env python
"""Step 05 — feature importance (RF / permutation / chi2) + incremental accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.models.feature_importance import run_feature_importance
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
    results = run_feature_importance(
        df,
        metrics_dir=cfg["metrics_dir"],
        figures_dir=cfg["figures_dir"],
        stem=stem,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
    )
    print("Top 10 permutation features:")
    print(results["permutation"].head(10).to_string(index=False))
    print("\nIncremental accuracy (permutation order):")
    print(
        results["incremental"][results["incremental"]["method"] == "Permutation"]
        .to_string(index=False)
    )
    print(f"\nMetrics -> {cfg['metrics_dir']}")
    print(f"Figure  -> {cfg['figures_dir'] / (stem + '_incremental_accuracy.png')}")


if __name__ == "__main__":
    main()
