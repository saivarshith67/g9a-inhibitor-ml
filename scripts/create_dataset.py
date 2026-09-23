#!/usr/bin/env python
"""Convenience shortcut for pipeline steps 02 + 03.

Assembles coordinate JSONs from cache if needed, engineers features, writes the
unbalanced interim CSV, then optionally balances (SMOTE / RUS) into processed/.

Equivalent to running:
  python scripts/02_build_features.py
  python scripts/03_balance.py

Examples:
  python scripts/create_dataset.py
  python scripts/create_dataset.py --balancing smote
  python scripts/create_dataset.py --balancing none
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.assemble_cache import ensure_coordinate_jsons
from g9a_ml.data.balance import balance_dataset, save_balanced
from g9a_ml.data.build_dataset import build_feature_table, save_unbalanced
from g9a_ml.paths import ensure_dirs, load_config, raw_dir_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--balancing",
        choices=["smote", "rus", "none"],
        default=None,
        help="Class balancing for the processed CSV (default: config value).",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Override processed filename stem (default: with_solubility_<balancing>).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = raw_dir_for(cfg, "with")
    sol = "with"

    print("=" * 60)
    print("CREATE DATASET")
    print("=" * 60)
    print(f"Raw dir: {raw}")

    # 1) Ensure coordinate JSONs (from cache if needed)
    print("\n[1/4] Ensuring coordinate JSON files...")
    coords = ensure_coordinate_jsons(raw)
    for name, path in coords.items():
        mb = path.stat().st_size / 1e6
        print(f"  OK  {name:30} {mb:8.2f} MB")

    # 2) Feature engineering
    print("\n[2/4] Building engineered feature table...")
    df = build_feature_table(raw, solubility=sol)
    print(f"  Shape: {df.shape}")
    print(f"  Target counts:\n{df['target'].value_counts().to_string()}")

    # 3) Save unbalanced
    unbalanced_path = cfg["interim_dir"] / f"{sol}_solubility_unbalanced.csv"
    print(f"\n[3/4] Saving unbalanced CSV -> {unbalanced_path}")
    save_unbalanced(df, unbalanced_path)

    # 4) Optional balancing
    balancing = args.balancing or cfg["dataset"]["balancing"]
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw),
        "n_rows_unbalanced": int(len(df)),
        "n_cols": int(df.shape[1]),
        "target_counts_unbalanced": {str(k): int(v) for k, v in df["target"].value_counts().items()},
        "feature_columns": [c for c in df.columns if c != "target"],
        "balancing": balancing,
        "unbalanced_csv": str(unbalanced_path),
    }

    if balancing == "none":
        out_path = cfg["processed_dir"] / (
            args.out_name or f"{sol}_solubility_unbalanced.csv"
        )
        save_unbalanced(df, out_path)
        meta["processed_csv"] = str(out_path)
        meta["n_rows_processed"] = int(len(df))
        meta["target_counts_processed"] = meta["target_counts_unbalanced"]
        print(f"\n[4/4] Balancing skipped. Copied to {out_path}")
    else:
        print(f"\n[4/4] Balancing with {balancing.upper()}...")
        rs = (
            cfg["dataset"]["smote_random_state"]
            if balancing == "smote"
            else cfg["dataset"]["random_state"]
        )
        balanced = balance_dataset(
            df,
            method=balancing,
            random_state=rs,
            majority_sample=cfg["dataset"].get("smote_majority_sample"),
        )
        stem = args.out_name or f"{sol}_solubility_{balancing}"
        out_path = cfg["processed_dir"] / f"{stem}.csv"
        save_balanced(balanced, out_path)
        meta["processed_csv"] = str(out_path)
        meta["n_rows_processed"] = int(len(balanced))
        meta["target_counts_processed"] = {
            str(k): int(v) for k, v in balanced["target"].value_counts().items()
        }
        print(f"  Shape: {balanced.shape}")
        print(f"  Target counts:\n{balanced['target'].value_counts().to_string()}")
        print(f"  Saved: {out_path}")

    meta_path = cfg["processed_dir"] / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Unbalanced : {unbalanced_path}")
    print(f"  Processed  : {out_path}")
    print(f"  Meta       : {meta_path}")
    print("=" * 60)
    print("\nNext: python scripts/04_eda_dataset.py")


if __name__ == "__main__":
    main()
