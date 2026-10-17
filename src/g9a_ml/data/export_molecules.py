"""Export SMILES + label table for structure-aware DL (no SMOTE)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from g9a_ml.data.loaders import load_core_targets, load_pubchem_computed

RDLogger.DisableLog("rdApp.*")


def _valid_smiles_mask(smiles: pd.Series) -> pd.Series:
    def _ok(s: object) -> bool:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return False
        text = str(s).strip()
        if not text:
            return False
        return Chem.MolFromSmiles(text) is not None

    return smiles.map(_ok)


def build_molecule_table(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build unbalanced molecule CSV aligned with the with-solubility cohort.

    Uses the same CID merge + uncharged filter as the classical feature path,
    without coordinate-derived engineered features or SMOTE.
    """
    t0, t1 = load_core_targets(raw_dir)
    p0, p1 = load_pubchem_computed(raw_dir)

    df0 = pd.merge(t0, p0, on="CID")
    df1 = pd.merge(t1, p1, on="CID")
    df = pd.concat([df0, df1], ignore_index=True)

    n_raw = len(df)
    df = df.dropna(subset=["SMILES", "XL"])
    df = df[df["Charge"] == 0].copy()
    n_after_charge = len(df)

    valid = _valid_smiles_mask(df["SMILES"])
    n_invalid = int((~valid).sum())
    df = df.loc[valid].copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

    keep = ["CID", "SID", "SMILES", "target", "Solubility_at_pH_7_4"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing from molecule table: {missing}")
    out = df[keep].drop_duplicates(subset=["CID"]).reset_index(drop=True)

    counts = {str(k): int(v) for k, v in out["target"].value_counts().sort_index().items()}
    meta: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "n_rows_merged": int(n_raw),
        "n_after_charge_filter": int(n_after_charge),
        "n_invalid_smiles": n_invalid,
        "n_rows": int(len(out)),
        "target_counts": counts,
        "balancing": "none",
        "note": "Real molecules only; no SMOTE. Class imbalance handled at training time.",
    }
    return out, meta


def save_molecule_table(
    df: pd.DataFrame,
    out_path: Path,
    meta: dict[str, Any],
    meta_path: Path | None = None,
) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    meta = dict(meta)
    meta["csv"] = str(out_path)
    mpath = meta_path or out_path.with_name("dataset_meta_mol.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_path, mpath
