"""Streamlit dashboard for the AIT403 Advanced Data Tutorial 1 analyses.

Run with:
    streamlit run app.py

The dashboard wraps the routines in ``analysis.py`` and adds interactive
controls, Plotly charts, result tables and CSV exports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_digits

from analysis import (
    DEFAULT_DIMENSIONS,
    DEFAULT_METRICS,
    DEFAULT_NOISE_COLUMNS,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_VARIANCE_THRESHOLDS,
    METRIC_LABELS,
    run_task1,
    run_task2,
    run_task3,
)

st.set_page_config(
    page_title="AIT403 · High-Dimensional Data Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 0.6rem;
        padding: 0.7rem 0.9rem;
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_task1(
    noise_columns: tuple[int, ...],
    n_neighbors: int,
    repeats: int,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    return run_task1(
        noise_columns=noise_columns,
        n_neighbors=n_neighbors,
        repeats=repeats,
        test_size=test_size,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def cached_task2(
    dimensions: tuple[int, ...],
    sample_sizes: tuple[int, ...],
    metrics: tuple[str, ...],
    n_queries: int,
    seed: int,
) -> pd.DataFrame:
    return run_task2(
        dimensions=dimensions,
        sample_sizes=sample_sizes,
        metrics=metrics,
        n_queries=n_queries,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def cached_task3(
    variance_thresholds: tuple[float, ...],
    correlation_threshold: float,
    random_trials: int,
    test_size: float,
    seed: int,
    n_estimators: int,
):
    return run_task3(
        variance_thresholds=variance_thresholds,
        correlation_threshold=correlation_threshold,
        random_trials=random_trials,
        test_size=test_size,
        seed=seed,
        n_estimators=n_estimators,
    )


@st.cache_data
def cached_digits():
    digits = load_digits()
    return digits.data, digits.target, digits.images


def csv_download(frame: pd.DataFrame, filename: str, key: str, label: str = "Download CSV"):
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.divider()
    st.subheader("Dataset")
    st.markdown(
        """
        **Digits** (scikit-learn)

        - 1,797 samples
        - 64 features (8×8 pixels)
        - 10 classes (digits 0–9)
        - Pixel values 0–16
        """
    )

    st.divider()
    if st.button("Clear results & cache", width="stretch"):
        st.cache_data.clear()
        for key in ("task1", "task2", "task3"):
            st.session_state.pop(key, None)
        st.rerun()

    with st.expander("Method notes"):
        st.markdown(
            """
            - KNN timing excludes a warm-up run; fit and predict are timed
              separately and averaged over repeats.
            - Task 2 averages contrast over multiple query points and reports
              the standard deviation.
            - Task 3 compares each selection method with a random subset of
              the **same size**, repeated several times.
            - Metrics are deterministic for a given seed and parameter set.
            """
        )

st.title("📊 High-Dimensional Data Dashboard")
st.caption(
    "AIT403 Advanced Data Analysis · Tutorial 1 · KNN under noise, distance "
    "concentration, and feature selection"
)

tab_overview, tab_task1, tab_task2, tab_task3 = st.tabs(
    [
        "Overview",
        "Task 1 · KNN & Noise",
        "Task 2 · Distance Concentration",
        "Task 3 · Feature Selection",
    ]
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    data, target, images = cached_digits()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples (N)", f"{data.shape[0]:,}")
    c2.metric("Features (D)", f"{data.shape[1]}")
    c3.metric("Classes", f"{len(np.unique(target))}")
    c4.metric("Pixel value range", "0–16")

    left, right = st.columns([1.1, 1.6], gap="large")

    with left:
        st.subheader("Class distribution")
        counts = (
            pd.Series(target, name="count")
            .value_counts()
            .sort_index()
            .rename_axis("digit")
            .reset_index()
        )
        fig = px.bar(
            counts,
            x="digit",
            y="count",
            text="count",
            labels={"digit": "Digit", "count": "Samples"},
        )
        fig.update_traces(marker_color="#4C78A8")
        fig.update_layout(showlegend=False, xaxis=dict(dtick=1))
        st.plotly_chart(style(fig, 330), width="stretch")

    with right:
        st.subheader("One sample per class")
        digit_columns = st.columns(10, gap="small")
        for digit, column in enumerate(digit_columns):
            index = int(np.where(target == digit)[0][0])
            with column:
                st.image(
                    images[index],
                    caption=str(digit),
                    clamp=True,
                    width="stretch",
                )

    st.divider()
    st.subheader("The three experiments")

    t1, t2, t3 = st.columns(3)
    with t1.container(border=True):
        st.markdown("#### 1 · KNN under noise")
        st.write(
            "Add up to 5,000 random features to the 64 real ones and track how "
            "accuracy, fit time, predict time and memory respond."
        )
    with t2.container(border=True):
        st.markdown("#### 2 · Distance concentration")
        st.write(
            "Measure how the contrast between nearest and farthest neighbours "
            "collapses as dimensionality grows, for L1 and L2 distances."
        )
    with t3.container(border=True):
        st.markdown("#### 3 · Feature selection")
        st.write(
            "Compare variance filtering, correlation filtering and random "
            "subsets with a Random Forest, using accuracy, time and memory."
        )

    with st.expander("What was improved compared with the original notebook"):
        st.markdown(
            """
            1. **Correct metadata** — `N` is the number of samples and `D` the
               number of features (the original printed `len(X)`/`len(y)` as
               both).
            2. **Reproducibility** — all noise, splits and subsets use explicit
               seeds.
            3. **Fair timing** — a warm-up run is discarded, `fit` and `predict`
               are timed separately, and results are averaged over repeats.
            4. **Consistent units** — Task 1 uses MB (train + test), Task 3 uses
               KB, with labels that match the numbers.
            5. **Stable distance contrast** — averaged over many query points
               with a reported standard deviation instead of one query.
            6. **Fair selection baseline** — random subsets are matched to the
               reduced feature count, and several variance thresholds are swept.
            7. **Interactive dashboard** — every parameter is adjustable, all
               results can be downloaded as CSV.
            """
        )


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------
with tab_task1:
    st.subheader("Task 1 · KNN accuracy vs. injected noise features")
    st.write(
        "The digits data starts with **64 features**. Noise columns are appended "
        "and a 5-nearest-neighbour classifier is re-trained. Timing excludes a "
        "warm-up run and is averaged over repeats."
    )

    with st.form("task1_form"):
        f1, f2, f3 = st.columns([2.2, 1, 1])
        noise_columns = f1.multiselect(
            "Noise columns to test",
            options=[0, 50, 200, 1000, 5000, 10000],
            default=list(DEFAULT_NOISE_COLUMNS),
            help="0 means the clean 64-feature baseline.",
        )
        n_neighbors = f2.slider("Neighbours (k)", 1, 15, 5)
        repeats = f3.slider("Timing repeats", 1, 10, 5)
        submitted = st.form_submit_button("Run benchmark", type="primary")

    if submitted:
        if not noise_columns:
            st.warning("Select at least one noise setting.")
        else:
            with st.spinner("Benchmarking KNN…"):
                st.session_state["task1"] = {
                    "data": cached_task1(
                        tuple(sorted(int(c) for c in noise_columns)),
                        int(n_neighbors),
                        int(repeats),
                        float(test_size),
                        int(seed),
                    ),
                    "params": {
                        "k": int(n_neighbors),
                        "repeats": int(repeats),
                        "test_size": float(test_size),
                        "seed": int(seed),
                    },
                }

    if "task1" not in st.session_state:
        st.info("Choose the settings above and press **Run benchmark**.")
    else:
        result = st.session_state["task1"]
        frame: pd.DataFrame = result["data"]
        params = result["params"]

        st.caption(
            f"k={params['k']} · repeats={params['repeats']} · "
            f"test={params['test_size']:.0%} · seed={params['seed']}"
        )

        baseline = frame.iloc[0]
        worst = frame.iloc[-1]
        delta = worst["accuracy_pct"] - baseline["accuracy_pct"]
        memory_ratio = worst["memory_mb"] / max(baseline["memory_mb"], 1e-12)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clean accuracy", f"{baseline['accuracy_pct']:.2f}%")
        m2.metric(
            f"Accuracy at {int(worst['noise_columns']):,} noise cols",
            f"{worst['accuracy_pct']:.2f}%",
            delta=f"{delta:+.2f} pp",
            delta_color="normal",
        )
        m3.metric("Features at max noise", f"{int(worst['n_features']):,}")
        m4.metric("Memory growth", f"{memory_ratio:.1f}×")

        if delta < -1:
            st.warning(
                f"Accuracy drops by {abs(delta):.2f} pp once "
                f"{int(worst['noise_columns']):,} noise features are added — the "
                "distance-based classifier is being diluted."
            )
        elif delta > 1:
            st.success(
                f"Accuracy actually improves by {delta:.2f} pp with noise in this "
                "run. KNN can benefit when extra dimensions spread out ties, but "
                "the result is data- and seed-dependent."
            )
        else:
            st.info(
                f"Accuracy changes by only {delta:+.2f} pp, but memory grows "
                f"{memory_ratio:.1f}× — the extra dimensions are not free."
            )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            fig = go.Figure(
                go.Scatter(
                    x=frame["noise_columns"],
                    y=frame["accuracy_pct"],
                    mode="lines+markers",
                    marker=dict(size=9, color="#4C78A8"),
                    line=dict(width=3),
                    error_y=dict(
                        type="data",
                        array=frame["accuracy_std_pct"],
                        visible=True,
                        color="#4C78A8",
                    ),
                    hovertemplate=(
                        "Noise: %{x:,}<br>Accuracy: %{y:.2f}%<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title="Accuracy vs. noise features",
                xaxis_title="Injected noise columns",
                yaxis_title="Accuracy (%)",
                showlegend=False,
            )
            st.plotly_chart(style(fig), width="stretch")

        with chart_right:
            timing = frame.melt(
                id_vars="noise_columns",
                value_vars=["fit_time_ms", "predict_time_ms"],
                var_name="stage",
                value_name="time_ms",
            )
            timing["stage"] = timing["stage"].map(
                {"fit_time_ms": "Fit", "predict_time_ms": "Predict"}
            )
            fig = px.line(
                timing,
                x="noise_columns",
                y="time_ms",
                color="stage",
                markers=True,
                log_y=True,
                labels={
                    "noise_columns": "Injected noise columns",
                    "time_ms": "Time (ms, log scale)",
                    "stage": "Stage",
                },
                title="Fit vs. predict time",
                color_discrete_sequence=["#F58518", "#54A24B"],
            )
            fig.update_traces(line_width=3, marker_size=8)
            st.plotly_chart(style(fig), width="stretch")

        mem_col, table_col = st.columns([1, 2], gap="large")
        with mem_col:
            fig = px.bar(
                frame,
                x="noise_columns",
                y="memory_mb",
                text=frame["memory_mb"].map(lambda value: f"{value:.2f} MB"),
                labels={
                    "noise_columns": "Injected noise columns",
                    "memory_mb": "Memory (MB)",
                },
                title="Memory (train + test)",
            )
            fig.update_traces(marker_color="#72B7B2", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 360), width="stretch")

        with table_col:
            display = frame[
                [
                    "noise_columns",
                    "n_features",
                    "accuracy_pct",
                    "accuracy_std_pct",
                    "macro_f1_pct",
                    "fit_time_ms",
                    "predict_time_ms",
                    "memory_mb",
                ]
            ].copy()
            display.columns = [
                "Noise cols",
                "Features",
                "Accuracy (%)",
                "Acc. σ",
                "Macro F1 (%)",
                "Fit (ms)",
                "Predict (ms)",
                "Memory (MB)",
            ]
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Acc. σ": st.column_config.NumberColumn(format="%.2f"),
                    "Macro F1 (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Fit (ms)": st.column_config.NumberColumn(format="%.3f"),
                    "Predict (ms)": st.column_config.NumberColumn(format="%.2f"),
                    "Memory (MB)": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            csv_download(frame, "task1_knn_noise.csv", "task1_download")


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------
with tab_task2:
    st.subheader("Task 2 · Distance concentration in high dimensions")
    st.write(
        "For each dimensionality **D**, random points are drawn and the contrast "
        "`(d_max − d_min) / d_min` is measured from several random queries. "
        "Contrast close to **0** means nearest and farthest neighbours are almost "
        "equally far away — distances stop being informative."
    )

    with st.form("task2_form"):
        c1, c2, c3 = st.columns([1.5, 1.2, 1])
        dimensions = c1.multiselect(
            "Dimensions (D)",
            options=[2, 10, 100, 1000, 10000],
            default=list(DEFAULT_DIMENSIONS),
            help="D=10,000 with N=5,000 allocates a large temporary matrix.",
        )
        sample_sizes = c2.multiselect(
            "Sample sizes (N)",
            options=[500, 1000, 5000, 10000],
            default=list(DEFAULT_SAMPLE_SIZES),
        )
        metrics = c3.multiselect(
            "Distance metric",
            options=list(METRIC_LABELS),
            default=list(DEFAULT_METRICS),
            format_func=lambda metric: METRIC_LABELS[metric],
        )
        n_queries = st.slider(
            "Query points to average",
            5,
            100,
            20,
            step=5,
            help="More queries give a more stable mean/std but take longer.",
        )
        submitted2 = st.form_submit_button("Run experiment", type="primary")

    if submitted2:
        if not dimensions or not sample_sizes or not metrics:
            st.warning("Select at least one dimension, sample size and metric.")
        else:
            with st.spinner("Computing distance contrasts…"):
                st.session_state["task2"] = {
                    "data": cached_task2(
                        tuple(sorted(int(d) for d in dimensions)),
                        tuple(sorted(int(n) for n in sample_sizes)),
                        tuple(sorted(metrics)),
                        int(n_queries),
                        int(seed),
                    ),
                    "queries": int(n_queries),
                    "seed": int(seed),
                }

    if "task2" not in st.session_state:
        st.info("Choose the settings above and press **Run experiment**.")
    else:
        result = st.session_state["task2"]
        frame: pd.DataFrame = result["data"]

        st.caption(
            f"{result['queries']} query points per cell · seed={result['seed']}"
        )

        l2 = frame[(frame["metric"] == "euclidean") & (frame["sample_size"] == 5000)]
        if l2.empty:
            l2 = frame[frame["metric"] == "euclidean"]
        if not l2.empty:
            first = l2.iloc[0]
            last = l2.iloc[-1]
            drop = (1 - last["contrast_mean"] / max(first["contrast_mean"], 1e-12)) * 100
            k1, k2, k3 = st.columns(3)
            k1.metric(
                f"Contrast at D={int(first['dimensions'])}",
                f"{first['contrast_mean']:.3f}",
            )
            k2.metric(
                f"Contrast at D={int(last['dimensions']):,}",
                f"{last['contrast_mean']:.3f}",
            )
            k3.metric("Contrast reduction", f"{drop:.2f}%")

        fig = go.Figure()
        palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
        for color, ((metric, sample_size), group) in zip(
            palette, frame.groupby(["metric", "sample_size"], sort=True)
        ):
            group = group.sort_values("dimensions")
            fig.add_trace(
                go.Scatter(
                    x=group["dimensions"],
                    y=group["contrast_mean"],
                    name=f"N={sample_size:,} · {METRIC_LABELS.get(metric, metric)}",
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(size=9),
                    error_y=dict(
                        type="data",
                        array=group["contrast_std"],
                        visible=True,
                        thickness=1.2,
                    ),
                    hovertemplate=(
                        "D=%{x:,}<br>Contrast=%{y:.3f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title="Distance contrast vs. dimensionality",
            xaxis_title="Dimensions (D, log scale)",
            yaxis_title="Contrast: (d_max − d_min) / d_min",
        )
        fig.update_xaxes(type="log", dtick=1)
        fig.update_yaxes(type="log", dtick=1)
        st.plotly_chart(style(fig, 440), width="stretch")

        pivot = frame.pivot_table(
            index="dimensions",
            columns=["metric", "sample_size"],
            values="contrast_mean",
        )
        pivot.columns = [
            f"N={sample_size:,} · {METRIC_LABELS.get(metric, metric)}"
            for metric, sample_size in pivot.columns
        ]
        pivot = pivot.reset_index().rename(columns={"dimensions": "D"})

        table_col, note_col = st.columns([2, 1], gap="large")
        with table_col:
            st.dataframe(
                pivot,
                width="stretch",
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.4f")
                    for column in pivot.columns
                    if column != "D"
                },
            )
            csv_download(frame, "task2_distance_contrast.csv", "task2_download")

        with note_col:
            st.markdown("#### Reading the result")
            st.markdown(
                """
                - A **high** contrast means the nearest neighbour is clearly
                  closer than the farthest one.
                - As **D** grows, contrast falls toward zero: all distances
                  concentrate, so “nearest” becomes less meaningful.
                - L1 (Manhattan) usually loses contrast a little more slowly
                  than L2 (Euclidean).
                - Error bars show one standard deviation across the random
                  query points.
                """
            )


# ---------------------------------------------------------------------------
# Task 3
# ---------------------------------------------------------------------------
with tab_task3:
    st.subheader("Task 3 · Feature selection with a Random Forest")
    st.write(
        "A 100-tree Random Forest is trained on the full 64 features, on "
        "variance-filtered and correlation-filtered subsets, and on random "
        "subsets matched to the same feature counts."
    )

    with st.form("task3_form"):
        c1, c2, c3, c4 = st.columns([1.8, 1.1, 1, 1])
        variance_thresholds = c1.multiselect(
            "Variance thresholds",
            options=[0.5, 1.0, 2.0, 5.0, 10.0, 22.0, 40.0],
            default=list(DEFAULT_VARIANCE_THRESHOLDS),
            help="Features with variance below the threshold are dropped.",
        )
        correlation_threshold = c2.slider(
            "Correlation cut-off",
            0.50,
            0.99,
            0.90,
            step=0.01,
            help="Drop one feature from each pair correlated above this value.",
        )
        random_trials = c3.slider("Random trials", 1, 20, 10)
        n_estimators = c4.select_slider(
            "Trees", options=[20, 50, 100, 200, 300], value=100
        )
        submitted3 = st.form_submit_button("Run comparison", type="primary")

    if submitted3:
        if not variance_thresholds:
            st.warning("Select at least one variance threshold.")
        else:
            with st.spinner("Training Random Forests… this can take a few seconds."):
                st.session_state["task3"] = {
                    "data": cached_task3(
                        tuple(sorted(float(t) for t in variance_thresholds)),
                        float(correlation_threshold),
                        int(random_trials),
                        float(test_size),
                        int(seed),
                        int(n_estimators),
                    ),
                    "params": {
                        "corr": float(correlation_threshold),
                        "trials": int(random_trials),
                        "trees": int(n_estimators),
                        "test_size": float(test_size),
                        "seed": int(seed),
                    },
                }

    if "task3" not in st.session_state:
        st.info("Choose the settings above and press **Run comparison**.")
    else:
        result = st.session_state["task3"]
        output = result["data"]
        params = result["params"]
        frame: pd.DataFrame = output.results
        random_frame: pd.DataFrame = output.random_baselines

        st.caption(
            f"{params['trees']} trees · correlation |r| > {params['corr']:.2f} · "
            f"{params['trials']} random trials · test={params['test_size']:.0%} · "
            f"seed={params['seed']}"
        )

        baseline = frame.iloc[0]
        candidates = frame.iloc[1:]
        if candidates.empty:
            best = baseline
        else:
            best = candidates.loc[candidates["accuracy_pct"].idxmax()]
        best_delta = best["accuracy_pct"] - baseline["accuracy_pct"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Baseline accuracy", f"{baseline['accuracy_pct']:.2f}%")
        k2.metric(
            "Best subset",
            f"{best['accuracy_pct']:.2f}%",
            delta=f"{best_delta:+.2f} pp",
        )
        k3.metric("Best feature count", f"{int(best['n_features'])} / 64")
        if not random_frame.empty:
            random_at_best = random_frame[
                random_frame["n_features"] == int(best["n_features"])
            ]
            if not random_at_best.empty:
                random_acc = float(random_at_best.iloc[0]["accuracy_pct"])
                k4.metric(
                    "Random subset, same size",
                    f"{random_acc:.2f}%",
                    delta=f"{best['accuracy_pct'] - random_acc:+.2f} pp vs random",
                )
            else:
                k4.metric("Random subsets", f"{len(random_frame)} sizes tested")

        # ---- Accuracy comparison -----------------------------------------
        plot_rows = []
        for _, row in frame.iterrows():
            plot_rows.append(
                {
                    "method": row["method"],
                    "family": row["family"],
                    "n_features": int(row["n_features"]),
                    "accuracy_pct": float(row["accuracy_pct"]),
                    "accuracy_std_pct": 0.0,
                }
            )
        for _, row in random_frame.iterrows():
            plot_rows.append(
                {
                    "method": f"Random subset ({int(row['n_features'])} features)",
                    "family": "Random",
                    "n_features": int(row["n_features"]),
                    "accuracy_pct": float(row["accuracy_pct"]),
                    "accuracy_std_pct": float(row["accuracy_std_pct"]),
                }
            )
        plot_frame = pd.DataFrame(plot_rows)

        fig = px.bar(
            plot_frame,
            x="accuracy_pct",
            y="method",
            orientation="h",
            color="family",
            error_x="accuracy_std_pct",
            text=plot_frame["accuracy_pct"].map(lambda value: f"{value:.1f}%"),
            labels={"accuracy_pct": "Accuracy (%)", "method": "", "family": "Method"},
            title="Accuracy by selection strategy",
            color_discrete_sequence=["#9E9E9E", "#4C78A8", "#54A24B", "#F58518"],
        )
        fig.add_vline(
            x=float(baseline["accuracy_pct"]),
            line_dash="dash",
            line_color="#E45756",
            annotation_text="Baseline",
            annotation_position="top",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 420), width="stretch")

        # ---- Accuracy vs. cost -------------------------------------------
        scatter_col, importance_col = st.columns(2, gap="large")
        with scatter_col:
            fig = px.scatter(
                plot_frame,
                x="n_features",
                y="accuracy_pct",
                color="family",
                size=plot_frame["accuracy_std_pct"].fillna(0) + 6,
                hover_name="method",
                labels={
                    "n_features": "Number of features",
                    "accuracy_pct": "Accuracy (%)",
                    "family": "Method",
                },
                title="Accuracy vs. feature count",
                color_discrete_sequence=["#9E9E9E", "#4C78A8", "#54A24B", "#F58518"],
            )
            fig.add_hline(
                y=float(baseline["accuracy_pct"]),
                line_dash="dot",
                line_color="#E45756",
            )
            st.plotly_chart(style(fig, 400), width="stretch")

        with importance_col:
            top = output.feature_importance.head(15).sort_values("importance")
            fig = px.bar(
                top,
                x="importance",
                y=top["feature"].astype(str),
                orientation="h",
                labels={"importance": "Importance", "y": "Pixel feature"},
                title="Top 15 features · baseline Random Forest",
            )
            fig.update_traces(marker_color="#72B7B2")
            fig.update_layout(yaxis_title="Feature index")
            st.plotly_chart(style(fig, 400), width="stretch")

        # ---- Tables ------------------------------------------------------
        table = frame[
            [
                "method",
                "n_features",
                "accuracy_pct",
                "macro_f1_pct",
                "train_time_ms",
                "memory_kb",
            ]
        ].copy()
        table["delta_pp"] = table["accuracy_pct"] - float(baseline["accuracy_pct"])
        table["speedup"] = float(baseline["train_time_ms"]) / table["train_time_ms"]
        table["memory_saved_pct"] = (
            1 - table["memory_kb"] / float(baseline["memory_kb"])
        ) * 100

        display = table[
            [
                "method",
                "n_features",
                "accuracy_pct",
                "delta_pp",
                "macro_f1_pct",
                "train_time_ms",
                "speedup",
                "memory_kb",
                "memory_saved_pct",
            ]
        ].copy()
        display.columns = [
            "Method",
            "Features",
            "Accuracy (%)",
            "Δ vs baseline (pp)",
            "Macro F1 (%)",
            "Train time (ms)",
            "Speed-up",
            "Memory (KB)",
            "Memory saved (%)",
        ]
        st.markdown("#### Selection results")
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "Accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
                "Δ vs baseline (pp)": st.column_config.NumberColumn(format="%+.2f"),
                "Macro F1 (%)": st.column_config.NumberColumn(format="%.2f"),
                "Train time (ms)": st.column_config.NumberColumn(format="%.1f"),
                "Speed-up": st.column_config.NumberColumn(format="%.2f×"),
                "Memory (KB)": st.column_config.NumberColumn(format="%.1f"),
                "Memory saved (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        csv_download(table, "task3_feature_selection.csv", "task3_download")

        if not random_frame.empty:
            st.markdown("#### Matched random baselines")
            random_display = random_frame.copy()
            random_display.columns = [
                "Features",
                "Trials",
                "Accuracy (%)",
                "Acc. σ",
                "Macro F1 (%)",
                "Train time (ms)",
                "Memory (KB)",
            ]
            st.dataframe(
                random_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Acc. σ": st.column_config.NumberColumn(format="%.2f"),
                    "Macro F1 (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Train time (ms)": st.column_config.NumberColumn(format="%.1f"),
                    "Memory (KB)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            csv_download(
                random_frame,
                "task3_random_baselines.csv",
                "task3_random_download",
            )

        with st.expander("How to read this comparison"):
            st.markdown(
                """
                - **Δ vs baseline** is the accuracy change in percentage points.
                - **Speed-up** compares training time with the full-feature
                  baseline; values above 1× are faster.
                - **Memory saved** covers the feature matrices (train + test),
                  not the Random Forest model itself.
                - A selection method is only useful if it beats the **random
                  subset of the same size**, not just the baseline.
                - Timings vary with machine load; accuracy is the stable signal.
                """
            )
