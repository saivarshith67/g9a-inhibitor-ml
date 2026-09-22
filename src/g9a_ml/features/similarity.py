"""SMILES Tanimoto similarity vs lysine (RDKit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

LYSINE_SMILES = "C(CCN)CC(C(=O)O)N"


def lysine_similarity(smiles: pd.Series) -> pd.Series:
    """Tanimoto similarity of each compound SMILES to lysine.

    Lysine is prepended as in the original notebook; the lysine self-match
    (first entry) is discarded.
    """
    values = list(smiles.values)
    values = [LYSINE_SMILES] + values

    mols = [Chem.MolFromSmiles(s) for s in values]
    fpgen = AllChem.GetRDKitFPGenerator()
    fps = []
    for mol in mols:
        if mol is None:
            fps.append(None)
        else:
            fps.append(fpgen.GetFingerprint(mol))

    lysine_fp = fps[0]
    sims = []
    for fp in fps:
        if fp is None or lysine_fp is None:
            sims.append(np.nan)
        else:
            sims.append(round(DataStructs.TanimotoSimilarity(lysine_fp, fp), 3))

    # Drop lysine self-similarity (index 0)
    return pd.Series(sims[1:], name="Similarity", dtype=float)
