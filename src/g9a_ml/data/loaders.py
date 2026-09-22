"""Load raw PubChem-derived tables from data/raw folders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_core_targets(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labeled CID/SID tables for target 0 and 1."""
    t0 = _read_csv(raw_dir / "data_for_one_column_target_0.csv")
    t1 = _read_csv(raw_dir / "data_for_one_column_target_1.csv")
    for df in (t0, t1):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    return t0, t1


def load_pubchem_computed(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = _read_csv(raw_dir / "pubChemComputed_target_0.csv")
    t1 = _read_csv(raw_dir / "pubChemComputed_target_1.csv")
    return t0, t1


def required_coord_files(raw_dir: Path, solubility: str = "with") -> dict[str, Path]:
    """Coordinate JSON files needed for feature engineering."""
    if solubility == "with":
        return {
            "sid_2d_0": raw_dir / "SID_2D_target_0.json",
            "sid_2d_1": raw_dir / "SID_2D_target_1.json",
            "cid_3d_0": raw_dir / "CID_3D_target_0.json",
            "cid_3d_1": raw_dir / "CID_3D_target_1.json",
        }
    # without solubility uses isomer-specific files in the original notebooks
    return {
        "sid_2d_0": raw_dir / "SID_2D_isomers_target_0.json",
        "sid_2d_1": raw_dir / "SID_2D_isomers_target_1.json",
        "cid_3d_0": raw_dir / "CID_3D_isomers_target_0.json",
        "cid_3d_1": raw_dir / "CID_3D_isomers_target_1.json",
    }


def missing_coord_files(raw_dir: Path, solubility: str = "with") -> list[Path]:
    return [p for p in required_coord_files(raw_dir, solubility).values() if not p.exists()]
