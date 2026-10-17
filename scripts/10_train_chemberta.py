#!/usr/bin/env python
"""Step 10 — full fine-tune ChemBERTa on SMILES (no SMOTE, no LoRA)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.metrics import save_metrics_json
from g9a_ml.models.chemberta import ChemBERTaConfig, train_chemberta_on_frame
from g9a_ml.paths import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--split",
        choices=["scaffold", "random", "both"],
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test: use first N rows.")
    parser.add_argument("--batch-size", type=int, default=None)
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

    craw = mol_cfg.get("chemberta", {})
    ccfg = ChemBERTaConfig(
        model_name=str(craw.get("model_name", "seyonec/ChemBERTa-zinc-base-v1")),
        max_length=int(craw.get("max_length", 128)),
        epochs=int(args.epochs or craw.get("epochs", 3)),
        batch_size=int(args.batch_size or craw.get("batch_size", 16)),
        lr=float(craw.get("lr", 2e-5)),
        weight_decay=float(craw.get("weight_decay", 0.01)),
        patience=int(craw.get("patience", 2)),
        seed=int(craw.get("seed", 42)),
    )

    if args.split == "both" or args.split is None:
        splits = list(mol_cfg.get("splits", ["scaffold", "random"]))
    else:
        splits = [args.split]

    model_root = cfg["models_dir"] / "chemberta"
    for split in splits:
        print(f"\n===== ChemBERTa full fine-tune / split={split} =====")
        result = train_chemberta_on_frame(
            df,
            split=split,  # type: ignore[arg-type]
            cfg=ccfg,
            smiles_col=mol_cfg.get("smiles_col", "SMILES"),
            target_col=mol_cfg.get("target_col", "target"),
            test_size=float(mol_cfg.get("test_size", 0.20)),
            val_size=float(mol_cfg.get("val_size", 0.10)),
            random_state=int(mol_cfg.get("random_state", 5)),
            out_dir=model_root,
        )
        metrics_path = cfg["metrics_dir"] / f"chemberta_{split}_metrics.json"
        save_metrics_json(result, metrics_path)
        print(f"Test metrics: {result['metrics']}")
        print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
