"""Train/val/test splits for molecule SMILES tables."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

SplitName = Literal["scaffold", "random"]


def _murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return smiles


def scaffold_split_indices(
    smiles: list[str] | pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices by Bemis–Murcko scaffold (group-aware).

    Scaffolds are sorted by size (largest first) and assigned greedily to the
    split that is farthest below its target fraction — a common cheminformatics
    practice to reduce structural leakage vs random split.
    """
    smiles_list = list(smiles)
    n = len(smiles_list)
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles_list):
        scaffolds.setdefault(_murcko_scaffold(str(s)), []).append(i)

    rng = np.random.RandomState(random_state)
    items = list(scaffolds.items())
    rng.shuffle(items)
    # Largest scaffolds first after seeded shuffle of equal sizes
    scaffold_groups = [idxs for _, idxs in sorted(items, key=lambda kv: len(kv[1]), reverse=True)]

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for group in scaffold_groups:
        if len(test_idx) + len(group) <= n_test:
            test_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    return (
        np.asarray(sorted(train_idx), dtype=int),
        np.asarray(sorted(val_idx), dtype=int),
        np.asarray(sorted(test_idx), dtype=int),
    )


def random_split_indices(
    n: int,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 5,
    stratify: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    val_ratio_of_trainval = val_size / (1.0 - test_size)
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    strat_tv = None if stratify is None else np.asarray(stratify)[trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_ratio_of_trainval,
        random_state=random_state,
        stratify=strat_tv,
    )
    return (
        np.asarray(sorted(train_idx), dtype=int),
        np.asarray(sorted(val_idx), dtype=int),
        np.asarray(sorted(test_idx), dtype=int),
    )


def split_molecule_frame(
    df: pd.DataFrame,
    split: SplitName = "scaffold",
    smiles_col: str = "SMILES",
    target_col: str = "target",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split == "scaffold":
        tr, va, te = scaffold_split_indices(
            df[smiles_col],
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    elif split == "random":
        tr, va, te = random_split_indices(
            len(df),
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify=df[target_col].values,
        )
    else:
        raise ValueError(f"Unknown split: {split}")
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
