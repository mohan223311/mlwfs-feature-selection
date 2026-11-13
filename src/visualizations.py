"""Matplotlib figures (return Figure objects so both CLI and Streamlit can use them)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def confusion_matrix_fig(cm, class_names, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    return fig


def correlation_heatmap_fig(X, feature_indices, title="Correlation of selected features"):
    sub = X.iloc[:, list(feature_indices)]
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(sub.columns) + 2),) * 2)
    sns.heatmap(sub.corr(), annot=len(sub.columns) <= 12, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def accuracy_bar_fig(table):
    base = table[table.experiment.str.startswith("Baseline")]
    sel = table[table.experiment.str.startswith("MLWFS")]
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, base.accuracy.values, 0.4, label="All features")
    ax.bar(x + 0.2, sel.accuracy.values, 0.4, label="MLWFS subset")
    ax.set_xticks(x, base.classifier.values)
    ax.set(ylabel="CV accuracy", title="Accuracy: all vs selected features",
           ylim=(max(0, min(base.accuracy.min(), sel.accuracy.min()) - 0.1), 1.0))
    ax.legend()
    fig.tight_layout()
    return fig


def feature_count_bar_fig(table):
    base = table[table.experiment.str.startswith("Baseline")]
    sel = table[table.experiment.str.startswith("MLWFS")]
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, base.n_features.values, 0.4, label="Original")
    ax.bar(x + 0.2, sel.n_features.values, 0.4, label="Selected")
    ax.set_xticks(x, base.classifier.values)
    ax.set(ylabel="# features", title="Number of features")
    ax.legend()
    fig.tight_layout()
    return fig


def sbs_curve_fig(result):
    h = result.sbs_history
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([s["n_features"] for s in h], [s["accuracy"] for s in h], marker="o")
    ax.invert_xaxis()
    ax.set(xlabel="# features remaining", ylabel="CV accuracy", title=f"SBS path ({result.classifier})")
    fig.tight_layout()
    return fig
