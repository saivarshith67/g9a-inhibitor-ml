#!/usr/bin/env python
"""Step 07 — train final 5-feature RandomForest (paper optimal model)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.models.train import depth_sweep, train_final_rf
from g9a_ml.paths import dataset_stem, ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--sweep-depth",
        action="store_true",
        help="Also sweep max_depth 1..22 and save overfitting gap table.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    stem = dataset_stem(cfg)
    path = cfg["processed_dir"] / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/03_balance.py (or create_dataset.py) first."
        )

    df = pd.read_csv(path)
    features = cfg["final_features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Final features missing from dataset: {missing}")

    result = train_final_rf(
        df,
        feature_cols=features,
        model_params=cfg["final_model"],
        test_size=cfg["train"]["test_size"],
        split_random_state=cfg["train"]["random_state"],
        models_dir=cfg["models_dir"],
    )

    out_json = cfg["metrics_dir"] / f"{stem}_final_model.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "features": result["features"],
                "model_params": cfg["final_model"],
            },
            f,
            indent=2,
        )
    report_path = cfg["metrics_dir"] / f"{stem}_final_classification_report.txt"
    report_path.write_text(result["classification_report"], encoding="utf-8")

    print("=== Final RandomForest (5 features) ===")
    print("Features:", features)
    print("Train/Test sizes:", result["n_train"], result["n_test"])
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(result["confusion_matrix"])
    print("\nClassification report:\n")
    print(result["classification_report"])
    print(f"Saved metrics: {out_json}")
    print(f"Saved model:   {cfg['models_dir'] / 'final_rf.joblib'}")

    if args.sweep_depth:
        sweep = depth_sweep(
            df,
            depths=range(1, 23),
            feature_cols=features,
            test_size=cfg["train"]["test_size"],
            random_state=cfg["train"]["random_state"],
            model_kwargs={"random_state": 1},
        )
        sweep_path = cfg["metrics_dir"] / f"{stem}_final_depth_sweep.csv"
        sweep.to_csv(sweep_path, index=False)
        gap = cfg["overfitting_gap"]
        ok = sweep[sweep["gap"] <= gap]
        print(f"\nDepths with train-test gap <= {gap}:")
        print(ok.to_string(index=False) if len(ok) else "  (none)")
        print(f"Saved depth sweep: {sweep_path}")


if __name__ == "__main__":
    main()
