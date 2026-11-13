"""Classifier factory. Scaling is part of the pipeline so it is fit per training fold."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSIFIERS = ("KNN", "SVM", "RF")
RANDOM_STATE = 42


def get_classifier(name: str):
    name = name.upper()
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE))
    if name in ("RF", "RANDOM FOREST"):
        # trees are scale-invariant, scaler kept for a uniform pipeline
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        )
    raise ValueError(f"Unknown classifier '{name}'. Choose from {CLASSIFIERS}.")
