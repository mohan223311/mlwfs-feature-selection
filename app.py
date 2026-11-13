"""Streamlit UI:  streamlit run app.py"""
import io
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import evaluation, mlwfs, preprocessing, visualizations as viz

st.set_page_config(page_title="MLWFS Feature Selection", layout="wide")


def show_fig(container, fig, filename):
    """Render a figure and offer a PNG download."""
    container.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    container.download_button("Download PNG", buf.getvalue(), filename, "image/png", key=filename)


@st.cache_data(show_spinner=False)
def time_one_eval(X, y, clf, folds):
    """Seconds for one cross-validated evaluation of ~10 features (used for the runtime estimate)."""
    t = time.perf_counter()
    evaluation.quick_accuracy(X, y, list(range(min(10, X.shape[1]))), clf, folds)
    return time.perf_counter() - t


def estimate_seconds(X, y, clf, group_size, trials, folds):
    n = X.shape[1]
    g = min(group_size, n)
    evals = trials * math.ceil(n / group_size) + g * (g + 1) // 2
    return evals * time_one_eval(X, y, clf, folds)


def fmt_time(sec):
    return f"{sec:.0f}s" if sec < 90 else f"{sec / 60:.1f} min"

st.title("MLWFS - Machine-Learning Wrapper Feature Selection")
st.caption("Stage 1: random feature-group search  ->  Stage 2: Sequential Backward Selection")

with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload CSV", type="csv")
    use_demo = st.checkbox("Use built-in Breast Cancer dataset", value=up is None)
    df = preprocessing.load_csv(up) if up is not None else (
        preprocessing.load_builtin_breast_cancer() if use_demo else None)
    target = None
    if df is not None:
        target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    st.header("2. Settings")
    clfs = st.multiselect("Classifier(s)", ["KNN", "SVM", "RF"], default=["KNN"])
    group_size = st.number_input("Group size", 2, 50, 10)
    trials = st.number_input("Number of trials", 1, 500, 20)
    folds = st.slider("CV folds", 3, 10, 10)
    seed = st.number_input("Random seed", 0, 9999, 42)
    nested = st.checkbox("Also run nested CV (leak-free, slower)")
    run = st.button("Run MLWFS", type="primary", disabled=df is None or not clfs)

if df is None:
    st.info("Upload a CSV or tick the demo dataset in the sidebar.")
    st.stop()

try:
    ds = preprocessing.prepare_dataset(df, target)
except Exception as e:
    st.error(str(e)); st.stop()

# ---- data validation: stop early with a clear message instead of a stack trace ----
import numpy as np
counts = np.bincount(ds.y)
if ds.X.shape[1] < 2:
    st.error("Need at least 2 usable numeric features after preprocessing."); st.stop()
if len(counts) < 2:
    st.error("The target column has only one class - pick a different target column."); st.stop()
if len(counts) > 20:
    st.error(f"Target '{target}' has {len(counts)} unique values - it looks like a continuous or ID "
             "column, not a class label. Pick a different target column."); st.stop()
if counts.min() < 2:
    bad = ds.class_names[int(counts.argmin())]
    st.error(f"Class '{bad}' has only {counts.min()} sample - every class needs at least 2 samples."); st.stop()
if counts.min() < folds:
    st.warning(f"Smallest class has {counts.min()} samples, so CV will automatically use {counts.min()} folds.")
if len(ds.y) < 30:
    st.warning("Very small dataset - results will be unreliable.")

stats = preprocessing.dataset_stats(ds)
c = st.columns(3)
c[0].metric("Samples", stats["samples"]); c[1].metric("Features", stats["features"]); c[2].metric("Classes", stats["classes"])
with st.expander("Data preview & class distribution"):
    st.dataframe(df.head(20))
    st.bar_chart(pd.Series(stats["class_distribution"]))

if clfs:
    est = {c: estimate_seconds(ds.X, ds.y, c, int(group_size), int(trials), folds) for c in clfs}
    msg = ", ".join(f"{c} ~{fmt_time(v)}" for c, v in est.items())
    total = sum(est.values())
    (st.warning if total > 180 else st.info)(
        f"Estimated MLWFS runtime: {msg} (total ~{fmt_time(total)}). "
        + ("Tip: lower the trials or skip RF for a quick demo." if total > 180 else ""))

if run:
    results, bar = {}, st.progress(0.0, "Starting...")
    for i, clf in enumerate(clfs):
        def cb(stage, frac, best, i=i, clf=clf):
            bar.progress((i + (0.7 * frac if stage == "stage1" else 0.7 + 0.3 * frac)) / len(clfs),
                         f"{clf}: {stage} - best acc {best:.3f}")
        results[clf] = mlwfs.run_mlwfs(ds.X, ds.y, clf, int(group_size), int(trials), folds, int(seed), cb)
    bar.progress(1.0, "Evaluating...")
    table = mlwfs.final_results_table(ds.X, ds.y, results, folds)
    nested_res = {}
    if nested:
        for clf in clfs:
            with st.spinner(f"Nested CV for {clf}..."):
                nested_res[clf] = mlwfs.nested_cv_mlwfs(ds.X, ds.y, clf, num_trials=min(int(trials), 20), seed=int(seed))
    bar.empty()
    st.session_state["out"] = (ds, results, table, nested_res, folds)

if "out" in st.session_state:
    ds, results, table, nested_res, folds = st.session_state["out"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results", "Selected features", "Plots", "Nested CV", "About"])
    with tab1:
        st.dataframe(table.drop(columns="features").round(4), width="stretch")
        st.download_button("Download results CSV", table.to_csv(index=False), "mlwfs_results.csv", "text/csv")
        st.caption("Note: these CV scores were also used to pick features, so they are optimistic. "
                   "See the Nested CV tab for an unbiased estimate.")
    with tab2:
        for clf, r in results.items():
            st.subheader(f"MLWFS-{clf}")
            st.write(f"Stage 1 group ({len(r.stage1_features)} features): acc {r.stage1_accuracy:.4f}")
            st.write(f"**Final subset ({len(r.selected_features)} features, acc {r.selected_accuracy:.4f}, "
                     f"{r.runtime_sec:.1f}s):** " + ", ".join(ds.X.columns[r.selected_features]))
            show_fig(st, viz.sbs_curve_fig(r), f"sbs_path_{clf}.png")
        if len(results) > 1:
            st.subheader("Feature overlap across classifiers")
            overlap = pd.DataFrame({c: [f in set(ds.X.columns[r.selected_features]) for f in ds.X.columns]
                                    for c, r in results.items()}, index=ds.X.columns)
            overlap["picked_by"] = overlap.sum(axis=1)
            overlap = overlap[overlap.picked_by > 0].sort_values("picked_by", ascending=False)
            st.dataframe(overlap.replace({True: "✔", False: ""}).assign(picked_by=overlap.picked_by),
                         width="stretch")
            st.caption("Features chosen by several classifiers are the most trustworthy.")
    with tab3:
        best = table[table.experiment.str.startswith("MLWFS")].sort_values("accuracy", ascending=False).iloc[0]
        r = results[best.classifier]
        a, b = st.columns(2)
        show_fig(a, viz.accuracy_bar_fig(table), "accuracy_comparison.png")
        show_fig(b, viz.feature_count_bar_fig(table), "feature_counts.png")
        show_fig(a, viz.correlation_heatmap_fig(ds.X, r.selected_features), "correlation_heatmap.png")
        show_fig(b, viz.confusion_matrix_fig(
            evaluation.cv_confusion_matrix(ds.X, ds.y, r.selected_features, best.classifier, folds),
            ds.class_names, f"Best model: MLWFS-{best.classifier}"), "confusion_matrix.png")
    with tab4:
        if not nested_res:
            st.info("Enable 'nested CV' in the sidebar and re-run.")
        for clf, nd in nested_res.items():
            st.subheader(clf)
            st.write(f"Selected-subset accuracy **{nd.acc_selected.mean():.4f}** vs all-features "
                     f"{nd.acc_all.mean():.4f}; avg {nd.n_selected.mean():.1f} features; "
                     f"avg runtime {nd.runtime_sec.mean():.1f}s/fold")
            st.dataframe(nd.drop(columns="selected_features").round(4))

    with tab5:
        st.markdown("""
    ### What is MLWFS?
    **Feature selection** finds the smallest set of columns that still predicts well. MLWFS is a *wrapper* method:
    it judges feature subsets by training a real classifier (KNN, SVM or Random Forest) on them.

    1. **Stage 1 - random group search:** shuffle the features, split into groups of N (default 10), score each
       group with cross-validation, repeat for many trials and keep the best group.
    2. **Stage 2 - Sequential Backward Selection:** from that group, repeatedly remove the feature whose removal
       hurts accuracy least, down to 1 feature; keep the best subset seen (ties go to the smaller subset).

    ### Classifiers
    - **KNN (k=3):** majority vote of the 3 most similar examples.
    - **SVM (rbf):** widest-margin curved decision boundary.
    - **Random Forest:** 100 decision trees voting (slower).

    ### Why nested CV?
    Plain CV scores are optimistic because the same folds choose the features *and* report accuracy. Nested CV runs
    the whole selection on each training split and scores the chosen features on a held-out fold - an honest estimate.
    Scaling is also fit only on training folds, so there is no leakage.
    """)
