#!/usr/bin/env python
"""Step 03 — balance classes with SMOTE or RUS and write processed CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.balance import balance_dataset, save_balanced
from g9a_ml.paths import dataset_stem, ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--balancing", choices=["smote", "rus"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.balancing:
        cfg["dataset"]["balancing"] = args.balancing

    sol = cfg["dataset"]["solubility"]
    src = cfg["interim_dir"] / f"{sol}_solubility_unbalanced.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run scripts/02_build_features.py first."
        )

    print(f"Loading: {src}")
    df = pd.read_csv(src)
    bal = cfg["dataset"]["balancing"]
    rs = (
        cfg["dataset"]["smote_random_state"]
        if bal == "smote"
        else cfg["dataset"]["random_state"]
    )
    out_df = balance_dataset(
        df,
        method=bal,
        random_state=rs,
        majority_sample=cfg["dataset"].get("smote_majority_sample"),
    )
    stem = dataset_stem(cfg)
    out = cfg["processed_dir"] / f"{stem}.csv"
    save_balanced(out_df, out)
    print(f"Balanced with {bal}: {out_df.shape}")
    print(f"Target counts:\n{out_df['target'].value_counts()}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
