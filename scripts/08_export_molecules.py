#!/usr/bin/env python
"""Step 08 — export SMILES + labels for structure-aware DL (no SMOTE)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.export_molecules import build_molecule_table, save_molecule_table
from g9a_ml.paths import ensure_dirs, load_config, raw_dir_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = raw_dir_for(cfg)

    mol_cfg = cfg.get("mol_dl", {})
    out_rel = mol_cfg.get("csv", "data/processed/mol_with_solubility.csv")
    out_path = Path(out_rel)
    if not out_path.is_absolute():
        out_path = cfg["_root"] / out_path

    print(f"Building molecule table from: {raw}")
    df, meta = build_molecule_table(raw)
    csv_path, meta_path = save_molecule_table(
        df,
        out_path,
        meta,
        meta_path=cfg["processed_dir"] / "dataset_meta_mol.json",
    )
    print(f"Shape: {df.shape}")
    print(f"Target counts:\n{df['target'].value_counts().to_string()}")
    print(f"Invalid SMILES dropped: {meta['n_invalid_smiles']}")
    print(f"Saved: {csv_path}")
    print(f"Meta:  {meta_path}")


if __name__ == "__main__":
    main()
