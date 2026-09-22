"""Class-balancing with SMOTE or RandomUnderSampler (paper Datasets 1–4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_dataset(
    df: pd.DataFrame,
    method: str = "smote",
    random_state: int = 0,
    majority_sample: int | None = None,
) -> pd.DataFrame:
    """Balance ``target``; drops ``Efficacy`` from features like the notebooks."""
    work = df.copy()
    if "Efficacy" in work.columns:
        X = work.drop(columns=["target", "Efficacy"])
    else:
        X = work.drop(columns=["target"])
    y = work["target"]

    if method == "smote" and majority_sample is not None:
        # Paper Dataset 2: subsample majority class before SMOTE
        maj = work[work["target"] == 0].sample(n=majority_sample, random_state=random_state)
        mino = work[work["target"] == 1]
        work = pd.concat([maj, mino], ignore_index=True)
        X = work.drop(columns=[c for c in ["target", "Efficacy"] if c in work.columns])
        y = work["target"]

    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "rus":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    X_res, y_res = sampler.fit_resample(X, y)
    out = X_res.copy()
    out["target"] = y_res.values
    out = out.sample(frac=1, random_state=1).reset_index(drop=True)
    return out


def save_balanced(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
