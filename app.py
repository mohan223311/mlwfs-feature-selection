"""Streamlit UI:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import evaluation, mlwfs, preprocessing, visualizations as viz

st.set_page_config(page_title="MLWFS Feature Selection", layout="wide")
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

stats = preprocessing.dataset_stats(ds)
c = st.columns(3)
c[0].metric("Samples", stats["samples"]); c[1].metric("Features", stats["features"]); c[2].metric("Classes", stats["classes"])
with st.expander("Data preview & class distribution"):
    st.dataframe(df.head(20))
    st.bar_chart(pd.Series(stats["class_distribution"]))

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
    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Selected features", "Plots", "Nested CV"])
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
            st.pyplot(viz.sbs_curve_fig(r))
    with tab3:
        best = table[table.experiment.str.startswith("MLWFS")].sort_values("accuracy", ascending=False).iloc[0]
        r = results[best.classifier]
        a, b = st.columns(2)
        a.pyplot(viz.accuracy_bar_fig(table)); b.pyplot(viz.feature_count_bar_fig(table))
        a.pyplot(viz.correlation_heatmap_fig(ds.X, r.selected_features))
        b.pyplot(viz.confusion_matrix_fig(
            evaluation.cv_confusion_matrix(ds.X, ds.y, r.selected_features, best.classifier, folds),
            ds.class_names, f"Best model: MLWFS-{best.classifier}"))
    with tab4:
        if not nested_res:
            st.info("Enable 'nested CV' in the sidebar and re-run.")
        for clf, nd in nested_res.items():
            st.subheader(clf)
            st.write(f"Selected-subset accuracy **{nd.acc_selected.mean():.4f}** vs all-features "
                     f"{nd.acc_all.mean():.4f}; avg {nd.n_selected.mean():.1f} features; "
                     f"avg runtime {nd.runtime_sec.mean():.1f}s/fold")
            st.dataframe(nd.drop(columns="selected_features").round(4))
