"""MLWFS: Machine-Learning-based Wrapper Feature Selection.

Stage 1: random feature-group search.  Stage 2: Sequential Backward Selection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evaluation import evaluate_features, quick_accuracy
from .models import get_classifier


@dataclass
class MLWFSResult:
    classifier: str
    stage1_features: list[int]
    stage1_accuracy: float
    selected_features: list[int]
    selected_accuracy: float
    sbs_history: list[dict] = field(default_factory=list)
    runtime_sec: float = 0.0


def stage1_random_groups(X, y, classifier, group_size=10, num_trials=100, cv_folds=10,
                         seed=42, progress=None):
    """Shuffle features, split into groups, keep the best-scoring group overall."""
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    est = get_classifier(classifier)
    best_acc, best_group = -1.0, []
    for t in range(num_trials):
        perm = rng.permutation(n)
        for s in range(0, n, group_size):
            group = sorted(perm[s:s + group_size].tolist())
            acc = quick_accuracy(X, y, group, est, cv_folds)
            if acc > best_acc:
                best_acc, best_group = acc, group
        if progress:
            progress("stage1", (t + 1) / num_trials, best_acc)
    return best_group, best_acc


def stage2_sbs(X, y, classifier, start_features, cv_folds=10, progress=None):
    """Sequential Backward Selection; returns best subset seen (ties -> smaller subset)."""
    est = get_classifier(classifier)
    current = list(start_features)
    acc = quick_accuracy(X, y, current, est, cv_folds)
    history = [{"n_features": len(current), "features": list(current), "accuracy": acc}]
    best_acc, best_set = acc, list(current)
    total = len(current) - 1
    while len(current) > 1:
        step_best_acc, step_best_set = -1.0, None
        for f in current:
            cand = [c for c in current if c != f]
            a = quick_accuracy(X, y, cand, est, cv_folds)
            if a > step_best_acc:
                step_best_acc, step_best_set = a, cand
        current = step_best_set
        history.append({"n_features": len(current), "features": list(current), "accuracy": step_best_acc})
        # equal accuracy -> prefer smaller subset (later steps are always smaller)
        if step_best_acc >= best_acc:
            best_acc, best_set = step_best_acc, list(current)
        if progress:
            progress("stage2", (total - len(current) + 1) / max(total, 1), best_acc)
    return best_set, best_acc, history


def run_mlwfs(X, y, classifier, group_size=10, num_trials=100, cv_folds=10, seed=42,
              progress=None) -> MLWFSResult:
    t0 = time.perf_counter()
    g, g_acc = stage1_random_groups(X, y, classifier, group_size, num_trials, cv_folds, seed, progress)
    sel, sel_acc, hist = stage2_sbs(X, y, classifier, g, cv_folds, progress)
    return MLWFSResult(classifier, g, g_acc, sel, sel_acc, hist, time.perf_counter() - t0)


def nested_cv_mlwfs(X, y, classifier, outer_folds=5, group_size=10, num_trials=20,
                    inner_folds=5, seed=42, progress=None) -> pd.DataFrame:
    """Leak-free estimate: run MLWFS on each outer-train split only, score on its held-out split."""
    from sklearn.model_selection import StratifiedKFold

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    est = get_classifier(classifier)
    rows = []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
        Xte, yte = X.iloc[te].reset_index(drop=True), y[te]
        res = run_mlwfs(Xtr, ytr, classifier, group_size, num_trials, inner_folds, seed + i)
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, f1_score
        full = clone(est).fit(Xtr, ytr).predict(Xte)
        sub = clone(est).fit(Xtr.iloc[:, res.selected_features], ytr).predict(Xte.iloc[:, res.selected_features])
        rows.append({
            "fold": i + 1,
            "n_selected": len(res.selected_features),
            "selected_features": res.selected_features,
            "acc_selected": accuracy_score(yte, sub),
            "f1_selected": f1_score(yte, sub, average="macro"),
            "acc_all": accuracy_score(yte, full),
            "f1_all": f1_score(yte, full, average="macro"),
            "runtime_sec": res.runtime_sec,
        })
        if progress:
            progress("nested", (i + 1) / outer_folds, rows[-1]["acc_selected"])
    return pd.DataFrame(rows)


def final_results_table(X, y, results: dict[str, MLWFSResult], cv_folds=10) -> pd.DataFrame:
    """Baseline vs MLWFS-<clf> summary (CV on the selected columns; see nested CV for unbiased numbers)."""
    rows = []
    for name in results:
        m = evaluate_features(X, y, range(X.shape[1]), name, cv_folds)
        rows.append({"experiment": f"Baseline-{name}", "classifier": name, "n_features": X.shape[1],
                     "accuracy": m["accuracy"], "f1": m["f1"], "runtime_sec": np.nan, "features": "all"})
    for name, r in results.items():
        m = evaluate_features(X, y, r.selected_features, name, cv_folds)
        rows.append({"experiment": f"MLWFS-{name}", "classifier": name, "n_features": len(r.selected_features),
                     "accuracy": m["accuracy"], "f1": m["f1"], "runtime_sec": r.runtime_sec,
                     "features": ", ".join(X.columns[r.selected_features])})
    return pd.DataFrame(rows)
