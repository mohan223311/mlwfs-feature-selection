import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from src import preprocessing, mlwfs, evaluation


def synth():
    X, y = make_classification(n_samples=400, n_features=30, n_informative=4, n_redundant=0,
                               shuffle=False, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)]); df["target"] = np.where(y == 1, "yes", "no")
    return preprocessing.prepare_dataset(df, "target")


def test_finds_informative_features():
    ds = synth()
    for clf in ("KNN", "SVM", "RF"):
        r = mlwfs.run_mlwfs(ds.X, ds.y, clf, 10, 15, 5, seed=1)
        names = set(ds.X.columns[r.selected_features])
        hits = names & {"f0", "f1", "f2", "f3"}
        print(clf, sorted(names), "acc", round(r.selected_accuracy, 3))
        assert len(hits) >= 2 and len(r.selected_features) < 30
        base = evaluation.evaluate_features(ds.X, ds.y, range(30), clf, 5)["accuracy"]
        assert r.selected_accuracy >= base - 0.02


def test_reproducible():
    ds = synth()
    a = mlwfs.run_mlwfs(ds.X, ds.y, "KNN", 10, 5, 5, seed=7)
    b = mlwfs.run_mlwfs(ds.X, ds.y, "KNN", 10, 5, 5, seed=7)
    assert a.selected_features == b.selected_features and a.stage1_features == b.stage1_features


def test_sbs_tie_prefers_smaller_and_history():
    ds = synth()
    r = mlwfs.run_mlwfs(ds.X, ds.y, "KNN", 10, 3, 5, seed=2)
    assert [h["n_features"] for h in r.sbs_history] == list(range(len(r.stage1_features), 0, -1))
    best = max(h["accuracy"] for h in r.sbs_history)
    assert r.selected_accuracy == best
    smallest = min(h["n_features"] for h in r.sbs_history if h["accuracy"] == best)
    assert len(r.selected_features) == smallest


def test_no_label_leak_random_labels_near_chance():
    ds = synth()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, len(ds.y))
    nd = mlwfs.nested_cv_mlwfs(ds.X, y, "KNN", outer_folds=5, num_trials=10, inner_folds=5, seed=0)
    print("random-label nested acc", nd.acc_selected.mean())
    assert nd.acc_selected.mean() < 0.62   # honest estimate ~0.5


def test_preprocessing_text_and_missing():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5, 6], "b": list("xyxyxy"), "c": [1]*6, "t": list("ppqqpq")})
    ds = preprocessing.prepare_dataset(df, "t")
    assert not ds.X.isna().any().any() and "c" not in ds.X.columns and set(ds.y) == {0, 1}


def test_metrics_keys():
    ds = synth()
    m = evaluation.evaluate_features(ds.X, ds.y, [0, 1], "SVM", 10)
    assert set(m) == {"accuracy", "precision", "recall", "f1"} and all(0 <= v <= 1 for v in m.values())
