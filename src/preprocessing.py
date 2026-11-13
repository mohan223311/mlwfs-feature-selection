"""Data loading and preprocessing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


@dataclass
class Dataset:
    X: pd.DataFrame          # numeric feature matrix (imputed, NOT scaled - scaling happens inside CV folds)
    y: np.ndarray            # integer-encoded labels
    feature_names: list[str]
    class_names: list[str]
    target_name: str


def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


def load_builtin_breast_cancer() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.data.copy()
    df["target"] = data.target_names[data.target]
    return df


def prepare_dataset(df: pd.DataFrame, target_col: str) -> Dataset:
    """Split df into X / y, impute missing values, encode labels.

    Scaling is deliberately NOT applied here: a StandardScaler fitted on the
    whole dataset would leak test-fold statistics. It lives inside the
    classifier pipeline (see models.py) and is fit per training fold.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # keep numeric columns only; one-hot encode low-cardinality text columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric):
        X = pd.get_dummies(X, columns=list(non_numeric), dtype=float)
    X = X.loc[:, X.nunique(dropna=True) > 1]  # drop constant columns

    imputed = SimpleImputer(strategy="median").fit_transform(X)
    X = pd.DataFrame(imputed, columns=X.columns)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    return Dataset(X, y, list(X.columns), [str(c) for c in le.classes_], target_col)


def dataset_stats(ds: Dataset) -> dict:
    counts = pd.Series(ds.y).value_counts().sort_index()
    return {
        "samples": len(ds.y),
        "features": ds.X.shape[1],
        "classes": len(ds.class_names),
        "class_distribution": {ds.class_names[i]: int(c) for i, c in counts.items()},
        "missing_values_imputed": 0,
    }
