"""A deliberately simple baseline detector.

The point of this project is the *data*, not the model. This exists so you can
measure whether a generated dataset is too easy, too hard or well balanced, and
so the Red-Team loop has something concrete to attack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ..features.build import model_matrix


def train_detector(features: pd.DataFrame, target: str = "is_laundering",
                   test_size: float = 0.3, seed: int = 0, max_iter: int = 300):
    X, y = model_matrix(features, target=target)
    idx = np.arange(len(y))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X.to_numpy(), y, idx, test_size=test_size, stratify=y, random_state=seed)
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, l2_regularization=1.0,
        early_stopping=True, random_state=seed).fit(Xtr, ytr)
    scores = model.predict_proba(Xte)[:, 1]
    return {
        "model": model,
        "feature_names": list(X.columns),
        "test_features": features.iloc[ite].reset_index(drop=True),
        "test_scores": scores,
        "train_index": itr,
        "test_index": ite,
        "X_test": Xte,
        "y_test": yte,
    }


def feature_importance(fit: dict, n_repeats: int = 5, sample: int = 3000,
                       seed: int = 0) -> pd.DataFrame:
    """Permutation importance - which behavioural signals the model actually used."""
    rng = np.random.default_rng(seed)
    X, y = fit["X_test"], fit["y_test"]
    if len(y) > sample:
        pick = rng.choice(len(y), size=sample, replace=False)
        X, y = X[pick], y[pick]
    r = permutation_importance(fit["model"], X, y, n_repeats=n_repeats,
                               random_state=seed, scoring="average_precision")
    return (pd.DataFrame({"feature": fit["feature_names"],
                          "importance": r.importances_mean.round(5),
                          "std": r.importances_std.round(5)})
            .sort_values("importance", ascending=False).reset_index(drop=True))
