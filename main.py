"""CLI runner:  python main.py --csv data/breast_cancer.csv --target target --classifiers KNN SVM RF"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src import evaluation, mlwfs, preprocessing, visualizations as viz

RESULTS = Path("results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/breast_cancer.csv")
    p.add_argument("--target", default="target")
    p.add_argument("--classifiers", nargs="+", default=["KNN"], choices=["KNN", "SVM", "RF"])
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--cv-folds", type=int, default=10)
    p.add_argument("--nested", action="store_true", help="also run nested CV (leak-free estimate)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    csv = Path(a.csv)
    if not csv.exists() and csv.name == "breast_cancer.csv":
        csv.parent.mkdir(exist_ok=True)
        preprocessing.load_builtin_breast_cancer().to_csv(csv, index=False)
    ds = preprocessing.prepare_dataset(preprocessing.load_csv(csv), a.target)
    print("Dataset:", preprocessing.dataset_stats(ds))
    print("\nBaseline (all features):\n", evaluation.baseline_table(ds.X, ds.y, a.cv_folds).round(4))

    results = {}
    for clf in a.classifiers:
        print(f"\n== MLWFS-{clf} ==")
        r = mlwfs.run_mlwfs(ds.X, ds.y, clf, a.group_size, a.trials, a.cv_folds, a.seed)
        results[clf] = r
        print(f" Stage 1 group ({len(r.stage1_features)}): acc={r.stage1_accuracy:.4f}")
        print(f" Stage 2 subset ({len(r.selected_features)}): acc={r.selected_accuracy:.4f}  "
              f"{[ds.feature_names[i] for i in r.selected_features]}  [{r.runtime_sec:.1f}s]")

    table = mlwfs.final_results_table(ds.X, ds.y, results, a.cv_folds)
    table.to_csv(RESULTS / "final_results.csv", index=False)
    print("\n", table.drop(columns="features").round(4))

    best = table[table.experiment.str.startswith("MLWFS")].sort_values("accuracy", ascending=False).iloc[0]
    r = results[best.classifier]
    figs = {
        "accuracy_comparison": viz.accuracy_bar_fig(table),
        "feature_counts": viz.feature_count_bar_fig(table),
        "correlation_heatmap": viz.correlation_heatmap_fig(ds.X, r.selected_features),
        "confusion_matrix": viz.confusion_matrix_fig(
            evaluation.cv_confusion_matrix(ds.X, ds.y, r.selected_features, best.classifier, a.cv_folds),
            ds.class_names, f"Confusion matrix - MLWFS-{best.classifier}"),
    }
    for name, fig in figs.items():
        fig.savefig(RESULTS / f"{name}.png", dpi=150)
        plt.close(fig)

    if a.nested:
        for clf in a.classifiers:
            nd = mlwfs.nested_cv_mlwfs(ds.X, ds.y, clf, num_trials=min(a.trials, 20), seed=a.seed)
            nd.to_csv(RESULTS / f"nested_cv_{clf}.csv", index=False)
            print(f"\nNested CV {clf}: selected acc={nd.acc_selected.mean():.4f} "
                  f"(all feats {nd.acc_all.mean():.4f}), avg #features={nd.n_selected.mean():.1f}")
    print(f"\nSaved outputs to {RESULTS}/")


if __name__ == "__main__":
    main()
