"""Build the engineered feature table (paper data-generation notebooks)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from g9a_ml.data.loaders import load_core_targets, load_pubchem_computed, missing_coord_files
from g9a_ml.features import (
    add_2d_volumes,
    add_3d_volumes,
    add_size_ratios,
    features_from_2d_json,
    features_from_3d_json,
    lysine_similarity,
    mass_proportions,
    relative_atom_proportions,
)

NEG_SHIFT_COLS = [
    "XL",
    "SX6",
    "SX",
    "SY6",
    "SY",
    "SX6_3D",
    "SX_3D",
    "SY6_3D",
    "SY_3D",
    "SZ6_3D",
    "SZ_3D",
    "XY_3D_volume",
    "XZ_3D_volume",
    "YZ_3D_volume",
    "C_rel_2D",
    "allAtoms_rel_2D",
    "Similarity",
]


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_feature_table(raw_dir: Path, solubility: str = "with") -> pd.DataFrame:
    """Reproduce feature engineering for the with-solubility datasets.

    Starts from the already-exported one-column target lists + PubChem computed
    properties + coordinate JSONs (avoids needing the huge AID 504332 dump).
    """
    missing = missing_coord_files(raw_dir, solubility="with")
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing coordinate files in {raw_dir}: {names}. "
            "Run: python scripts/01_download_coordinates.py"
        )

    t0, t1 = load_core_targets(raw_dir)
    p0, p1 = load_pubchem_computed(raw_dir)

    df0 = pd.merge(t0, p0, on="CID")
    df1 = pd.merge(t1, p1, on="CID")
    df = pd.concat([df0, df1], ignore_index=True)
    df = df.dropna(subset=["XL"])
    df = df[df["Charge"] == 0].drop(columns=["Charge"])

    feats_2d = pd.concat(
        [
            features_from_2d_json(_load_json(raw_dir / "SID_2D_target_0.json")),
            features_from_2d_json(_load_json(raw_dir / "SID_2D_target_1.json")),
        ],
        ignore_index=True,
    )
    df = pd.merge(df, feats_2d, on="SID")
    df = add_2d_volumes(df)

    feats_3d = pd.concat(
        [
            features_from_3d_json(_load_json(raw_dir / "CID_3D_target_0.json")),
            features_from_3d_json(_load_json(raw_dir / "CID_3D_target_1.json")),
        ],
        ignore_index=True,
    )
    df = pd.merge(df, feats_3d, on="CID")
    df = add_3d_volumes(df)
    df = df.dropna(subset=["SZ6_3D", "XZ_3D_volume", "YZ_3D_volume"]).reset_index(drop=True)

    rel = relative_atom_proportions(df["MF"].tolist())
    rel = rel.reset_index().rename(columns={"index": "row_id"})
    mass = mass_proportions(df["MF"].tolist())
    mass = mass.reset_index().rename(columns={"index": "row_id"})
    df = df.reset_index(drop=True).reset_index().rename(columns={"index": "row_id"})
    df = df.merge(rel, on=["row_id", "MF"])
    df = df.merge(mass, on=["row_id", "MF"])

    df = add_size_ratios(df)
    df["Similarity"] = lysine_similarity(df["SMILES"]).values

    present_neg = [c for c in NEG_SHIFT_COLS if c in df.columns]
    df[present_neg] = df[present_neg].add(20)

    drop_cols = [c for c in ["SID", "CID", "MF", "SMILES", "row_id"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def save_unbalanced(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
