#!/usr/bin/env python
"""Step 02 — engineer features from PubChem tables + coordinate JSONs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.build_dataset import build_feature_table, save_unbalanced
from g9a_ml.paths import ensure_dirs, load_config, raw_dir_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = raw_dir_for(cfg)
    sol = cfg["dataset"]["solubility"]

    print(f"Building features from: {raw}")
    df = build_feature_table(raw, solubility=sol)
    out = cfg["interim_dir"] / f"{sol}_solubility_unbalanced.csv"
    save_unbalanced(df, out)
    print(f"Shape: {df.shape}")
    print(f"Target counts:\n{df['target'].value_counts()}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
