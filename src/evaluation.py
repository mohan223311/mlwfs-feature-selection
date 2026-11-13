"""Evaluation helpers: stratified k-fold metrics on selected columns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .models import RANDOM_STATE, get_classifier


def _cols(X, feature_indices):
    return X.iloc[:, list(feature_indices)] if isinstance(X, pd.DataFrame) else X[:, list(feature_indices)]


def _skf(y, cv_folds):
    # never ask for more folds than the smallest class can support
    k = int(min(cv_folds, np.bincount(y).min()))
    if k < 2:
        raise ValueError("Smallest class has fewer than 2 samples; cannot stratify.")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)


def quick_accuracy(X, y, feature_indices, classifier, cv_folds=10) -> float:
    """Mean CV accuracy only - used inside the MLWFS search loops."""
    est = classifier if not isinstance(classifier, str) else get_classifier(classifier)
    res = cross_validate(clone(est), _cols(X, feature_indices), y, cv=_skf(y, cv_folds), scoring="accuracy")
    return float(res["test_score"].mean())


def evaluate_features(X, y, feature_indices, classifier, cv_folds=10) -> dict:
    """Stratified k-fold CV on the selected columns -> accuracy/precision/recall/F1 (macro)."""
    est = classifier if not isinstance(classifier, str) else get_classifier(classifier)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }
    res = cross_validate(clone(est), _cols(X, feature_indices), y, cv=_skf(y, cv_folds), scoring=scoring)
    return {m: float(res[f"test_{m}"].mean()) for m in scoring}


def cv_confusion_matrix(X, y, feature_indices, classifier, cv_folds=10):
    est = classifier if not isinstance(classifier, str) else get_classifier(classifier)
    pred = cross_val_predict(clone(est), _cols(X, feature_indices), y, cv=_skf(y, cv_folds))
    return confusion_matrix(y, pred)


def baseline_table(X, y, cv_folds=10) -> pd.DataFrame:
    rows = []
    for name in ("KNN", "SVM", "RF"):
        m = evaluate_features(X, y, range(X.shape[1]), name, cv_folds)
        rows.append({"classifier": name, "features": X.shape[1], **m})
    return pd.DataFrame(rows)
