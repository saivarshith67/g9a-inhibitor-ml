"""Train / evaluate classifiers and overfitting sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from g9a_ml.metrics import binary_metrics, metrics_table, save_metrics
from g9a_ml.models.classifiers import default_classifiers


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target"]
    X = df.drop(columns=["target"])
    if feature_cols is not None:
        X = X[feature_cols]
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    X_train_u, X_test_u, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_u)
    X_test = scaler.transform(X_test_u)
    return X_train, X_test, y_train, y_test, scaler


def compare_classifiers(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 5,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )
    clfs = default_classifiers()

    holdout_rows = []
    for name, model in clfs.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = binary_metrics(y_test, pred)
        holdout_rows.append({"algorithm": name, **m})

    # Scale inside each CV fold (avoids leakage; original notebooks scored unscaled X)
    cv_rows = []
    for name, model in default_classifiers().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring="accuracy")
        cv_rows.append(
            {
                "algorithm": name,
                "mean_cv_accuracy": round(float(scores.mean()), 4),
                "std_cv_accuracy": round(float(scores.std()), 4),
                "cv_scores": np.round(scores, 4).tolist(),
            }
        )

    return metrics_table(holdout_rows), metrics_table(cv_rows)


def depth_sweep(
    df: pd.DataFrame,
    depths: range | list[int],
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 5,
    model_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X, y = prepare_xy(df, feature_cols=feature_cols)
    X_train, X_test, y_train, y_test, _ = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )
    kwargs = dict(model_kwargs or {})
    rows = []
    for depth in depths:
        model = RandomForestClassifier(max_depth=depth, **kwargs)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        rows.append(
            {
                "max_depth": depth,
                "train_accuracy": round(float(train_acc), 4),
                "test_accuracy": round(float(test_acc), 4),
                "gap": round(float(train_acc - test_acc), 4),
            }
        )
    return pd.DataFrame(rows)


def train_final_rf(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict[str, Any],
    test_size: float = 0.2,
    split_random_state: int = 5,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    from g9a_ml.metrics import confusion_and_report

    X, y = prepare_xy(df, feature_cols=feature_cols)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, test_size=test_size, random_state=split_random_state
    )
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = binary_metrics(y_test, pred)
    cm, report = confusion_and_report(y_test, pred)

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / "final_rf.joblib")
        joblib.dump(scaler, models_dir / "final_scaler.joblib")
        joblib.dump(feature_cols, models_dir / "final_features.joblib")

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": feature_cols,
    }


def run_compare_and_save(
    df: pd.DataFrame,
    metrics_dir: Path,
    stem: str,
    test_size: float = 0.2,
    random_state: int = 5,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout, cv = compare_classifiers(
        df, test_size=test_size, random_state=random_state, cv_folds=cv_folds
    )
    holdout = holdout.sort_values("accuracy", ascending=False)
    cv = cv.sort_values("mean_cv_accuracy", ascending=False)
    save_metrics(holdout, metrics_dir / f"{stem}_holdout_metrics.csv")
    save_metrics(cv, metrics_dir / f"{stem}_cv_metrics.csv")
    return holdout, cv
