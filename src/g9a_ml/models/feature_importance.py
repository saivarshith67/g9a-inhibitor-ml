"""Feature importance analysis (RF, permutation, chi2/SelectKBest)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from g9a_ml.models.train import prepare_xy, split_and_scale


def rf_feature_importance(model: RandomForestClassifier, columns: list[str], top_n: int = 15) -> pd.Series:
    return pd.Series(model.feature_importances_, index=columns).nlargest(top_n)


def permutation_feature_importance(model, X_test, y_test, columns: list[str]) -> pd.Series:
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    return pd.Series(result.importances_mean, index=columns).sort_values(ascending=False)


def chi2_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    # chi2 requires non-negative features; shift if needed
    X_pos = X.copy()
    mins = X_pos.min()
    shift = mins.clip(upper=0).abs()
    X_pos = X_pos + shift
    selector = SelectKBest(chi2, k="all")
    selector.fit(X_pos, y)
    return pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)


def incremental_accuracy(
    df: pd.DataFrame,
    ordered_features: list[str],
    max_features: int = 11,
    test_size: float = 0.2,
    random_state: int = 5,
) -> pd.DataFrame:
    """Accuracy when adding features one-by-one in importance order (paper Fig. 3)."""
    rows = []
    for k in range(1, min(max_features, len(ordered_features)) + 1):
        feats = ordered_features[:k]
        X, y = prepare_xy(df, feature_cols=feats)
        X_train, X_test, y_train, y_test, _ = split_and_scale(
            X, y, test_size=test_size, random_state=random_state
        )
        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"n_features": k, "features": ",".join(feats), "accuracy": round(float(acc), 4)})
    return pd.DataFrame(rows)


def run_feature_importance(
    df: pd.DataFrame,
    metrics_dir: Path,
    figures_dir: Path,
    stem: str,
    test_size: float = 0.2,
    random_state: int = 5,
) -> dict[str, pd.DataFrame]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestClassifier(random_state=1, max_depth=11)
    model.fit(X_train, y_train)

    rf_imp = rf_feature_importance(model, list(X.columns), top_n=20)
    perm_imp = permutation_feature_importance(model, X_test, y_test, list(X.columns))
    chi_imp = chi2_feature_importance(X, y)

    rf_df = rf_imp.reset_index()
    rf_df.columns = ["feature", "importance"]
    perm_df = perm_imp.reset_index()
    perm_df.columns = ["feature", "importance"]
    chi_df = chi_imp.reset_index()
    chi_df.columns = ["feature", "score"]

    rf_df.to_csv(metrics_dir / f"{stem}_rf_importance.csv", index=False)
    perm_df.to_csv(metrics_dir / f"{stem}_permutation_importance.csv", index=False)
    chi_df.to_csv(metrics_dir / f"{stem}_chi2_importance.csv", index=False)

    # Incremental accuracy for top methods
    incr_frames = []
    for method, series in (
        ("RF importance", rf_imp),
        ("Permutation", perm_imp),
        ("Chi2", chi_imp),
    ):
        ordered = list(series.index)
        incr = incremental_accuracy(df, ordered, max_features=11, test_size=test_size, random_state=random_state)
        incr["method"] = method
        incr_frames.append(incr)
    incr_all = pd.concat(incr_frames, ignore_index=True)
    incr_all.to_csv(metrics_dir / f"{stem}_incremental_accuracy.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=incr_all, x="n_features", y="accuracy", hue="method", marker="o")
    plt.title("Accuracy vs number of features")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{stem}_incremental_accuracy.png", dpi=150)
    plt.close()

    return {
        "rf": rf_df,
        "permutation": perm_df,
        "chi2": chi_df,
        "incremental": incr_all,
    }
