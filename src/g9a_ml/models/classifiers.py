"""Classifier zoo matching the paper (scikit-learn + XGBoost)."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def default_classifiers(random_state: int = 42) -> dict:
    return {
        # probability=False: faster; metrics use hard predictions (as in the notebooks)
        "SVM": SVC(kernel="rbf", C=1, random_state=random_state),
        "Decision": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoost": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(verbosity=0, random_state=random_state),
    }
