#!/usr/bin/env python
"""Step 09 — train Chemprop-style D-MPNN on SMILES (no SMOTE).

Uses the in-repo directed message-passing implementation
(``g9a_ml.models.dmpnn``), which follows the Chemprop D-MPNN idea without
depending on the chemprop v1/v2 CLI API.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.metrics import save_metrics_json
from g9a_ml.models.dmpnn import DMPNNConfig, train_dmpnn_on_frame
from g9a_ml.paths import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--split",
        choices=["scaffold", "random", "both"],
        default=None,
        help="Which split(s) to run (default: config mol_dl.splits).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test: use first N rows.")
    parser.add_argument("--use-solubility", action="store_true", help="Append solubility as extra feature.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    mol_cfg = cfg.get("mol_dl", {})
    csv_rel = mol_cfg.get("csv", "data/processed/mol_with_solubility.csv")
    csv_path = Path(csv_rel) if Path(csv_rel).is_absolute() else cfg["_root"] / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run scripts/08_export_molecules.py first.")

    df = pd.read_csv(csv_path)
    if args.limit is not None:
        # Stratified sample so tiny smoke-tests keep both classes
        target_col = mol_cfg.get("target_col", "target")
        n = min(args.limit, len(df))
        parts = []
        for _, g in df.groupby(target_col):
            k = max(1, int(round(len(g) * (n / len(df)))))
            parts.append(g.sample(n=min(k, len(g)), random_state=0))
        df = pd.concat(parts, ignore_index=True)
        if len(df) > n:
            df = df.sample(n=n, random_state=0).reset_index(drop=True)
        print(f"Smoke-test limit: {len(df)} rows\n{df[target_col].value_counts().to_string()}")

    dcfg_raw = mol_cfg.get("dmpnn", {})
    dcfg = DMPNNConfig(
        hidden=int(dcfg_raw.get("hidden", 128)),
        depth=int(dcfg_raw.get("depth", 3)),
        dropout=float(dcfg_raw.get("dropout", 0.05)),
        ffn_hidden=int(dcfg_raw.get("ffn_hidden", 128)),
        epochs=int(args.epochs or dcfg_raw.get("epochs", 20)),
        batch_size=int(dcfg_raw.get("batch_size", 64)),
        lr=float(dcfg_raw.get("lr", 1e-4)),
        use_solubility=bool(args.use_solubility or dcfg_raw.get("use_solubility", False)),
        patience=int(dcfg_raw.get("patience", 5)),
    )

    if args.split == "both" or args.split is None:
        splits = list(mol_cfg.get("splits", ["scaffold", "random"]))
    else:
        splits = [args.split]

    model_dir = cfg["models_dir"] / "chemprop"
    model_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n===== D-MPNN / split={split} =====")
        result = train_dmpnn_on_frame(
            df,
            split=split,  # type: ignore[arg-type]
            cfg=dcfg,
            smiles_col=mol_cfg.get("smiles_col", "SMILES"),
            target_col=mol_cfg.get("target_col", "target"),
            test_size=float(mol_cfg.get("test_size", 0.20)),
            val_size=float(mol_cfg.get("val_size", 0.10)),
            random_state=int(mol_cfg.get("random_state", 5)),
            out_dir=model_dir,
        )
        metrics_path = cfg["metrics_dir"] / f"chemprop_{split}_metrics.json"
        save_metrics_json(result, metrics_path)
        # keep a copy of the model-dir metrics under outputs/metrics
        src = model_dir / f"dmpnn_{split}_metrics.json"
        if src.exists() and src.resolve() != metrics_path.resolve():
            shutil.copy2(src, metrics_path)
        print(f"Test metrics: {result['metrics']}")
        print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
