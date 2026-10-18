"""Core analysis routines for the AIT403 Advanced Data Tutorial 1 dashboard.

This module extracts the logic from ``Tutorial1.ipynb`` / ``Tutorial1_fixed.ipynb``
into reusable, testable functions with a few deliberate improvements:

* **Correct metadata**: ``N`` is the number of samples, ``D`` the number of features.
* **Reproducibility**: every random step uses an explicit seed.
* **Fair timing**: KNN is warmed up before timing; fit and predict are measured
  separately and averaged over repeats.
* **Consistent units**: memory is reported in MB for Task 1 and KB for Task 3.
* **Robust distance concentration**: contrast is averaged over many random
  queries instead of a single query point, and spread is reported.
* **Fair feature selection**: variance thresholds are swept, the correlation
  filter uses a configurable cut-off, and random baselines are matched to each
  reduced feature count.

Run ``python analysis.py`` to reproduce all three tasks in the terminal.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DEFAULT_SEED = 42
DEFAULT_NOISE_COLUMNS: tuple[int, ...] = (0, 50, 200, 1000, 5000)
DEFAULT_DIMENSIONS: tuple[int, ...] = (2, 10, 100, 1000, 10000)
DEFAULT_SAMPLE_SIZES: tuple[int, ...] = (500, 5000)
DEFAULT_METRICS: tuple[str, ...] = ("euclidean", "cityblock")
DEFAULT_VARIANCE_THRESHOLDS: tuple[float, ...] = (1.0, 5.0, 10.0, 22.0)

METRIC_LABELS = {
    "euclidean": "L2 (Euclidean)",
    "cityblock": "L1 (Manhattan)",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_digits_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the 8x8 digits dataset as ``(X, y)``: 1797 samples, 64 features."""
    digits = load_digits()
    return digits.data, digits.target


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    stratify: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/test, optionally preserving the class distribution."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if stratify else None,
    )


def add_noise_columns(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_columns: int,
    seed: int = DEFAULT_SEED,
    low: float = 0.0,
    high: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Append ``n_columns`` uniform-noise features drawn from ``[low, high)``.

    The range matches the original pixel scale (0-16) so the added features are
    not trivially separable from the real ones. ``n_columns <= 0`` is a no-op.
    """
    if n_columns <= 0:
        return X_train.copy(), X_test.copy()

    rng = np.random.default_rng(seed)
    noise_train = rng.uniform(low, high, size=(X_train.shape[0], n_columns))
    noise_test = rng.uniform(low, high, size=(X_test.shape[0], n_columns))
    return (
        np.hstack((X_train, noise_train)),
        np.hstack((X_test, noise_test)),
    )


# ---------------------------------------------------------------------------
# Task 1: KNN accuracy vs. injected noise features
# ---------------------------------------------------------------------------
def benchmark_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 5,
    repeats: int = 5,
) -> dict[str, float]:
    """Time KNN fit/predict and measure accuracy.

    A warm-up run (excluded from timing) removes first-call overhead such as
    lazy imports and BLAS initialisation. Each repeat trains a fresh model.
    """
    warm = KNeighborsClassifier(n_neighbors=n_neighbors)
    warm.fit(X_train, y_train)
    warm.predict(X_test)

    fit_times, predict_times, accuracies, f1_scores = [], [], [], []
    for _ in range(repeats):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_times.append(time.perf_counter() - start)

        accuracies.append(accuracy_score(y_test, predictions) * 100.0)
        f1_scores.append(f1_score(y_test, predictions, average="macro") * 100.0)

    return {
        "accuracy_pct": float(np.mean(accuracies)),
        "accuracy_std_pct": float(np.std(accuracies, ddof=1)) if repeats > 1 else 0.0,
        "macro_f1_pct": float(np.mean(f1_scores)),
        "fit_time_ms": float(np.mean(fit_times) * 1000.0),
        "predict_time_ms": float(np.mean(predict_times) * 1000.0),
    }


def run_task1(
    noise_columns: Sequence[int] = DEFAULT_NOISE_COLUMNS,
    n_neighbors: int = 5,
    repeats: int = 5,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Benchmark KNN on the digits data with increasing numbers of noise features.

    Returns one row per noise setting with accuracy, macro-F1, fit/predict time
    (ms) and data memory (MB, train + test).
    """
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, seed=seed)

    rows = []
    for n_columns in sorted({int(c) for c in noise_columns}):
        X_train_noisy, X_test_noisy = add_noise_columns(
            X_train, X_test, n_columns, seed=seed
        )
        metrics = benchmark_knn(
            X_train_noisy,
            X_test_noisy,
            y_train,
            y_test,
            n_neighbors=n_neighbors,
            repeats=repeats,
        )
        rows.append(
            {
                "noise_columns": n_columns,
                "n_features": int(X_train_noisy.shape[1]),
                "n_train": int(X_train_noisy.shape[0]),
                "n_test": int(X_test_noisy.shape[0]),
                **metrics,
                "memory_mb": (X_train_noisy.nbytes + X_test_noisy.nbytes) / 1024**2,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 2: distance concentration in high dimensions
# ---------------------------------------------------------------------------
def compute_contrast(
    sample_size: int,
    dimensions: Sequence[int],
    metric: str = "euclidean",
    n_queries: int = 20,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Average distance contrast ``(d_max - d_min) / d_min`` per dimension.

    Contrast is measured for ``n_queries`` random query points against
    ``sample_size`` random points. Averaging removes single-query noise; the
    standard deviation across queries is reported as an uncertainty estimate.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for dimension in sorted({int(d) for d in dimensions}):
        points = rng.random((sample_size, dimension))
        queries = rng.random((n_queries, dimension))

        distances = cdist(queries, points, metric=metric)
        d_min = distances.min(axis=1)
        d_max = distances.max(axis=1)
        contrast = (d_max - d_min) / d_min

        rows.append(
            {
                "sample_size": int(sample_size),
                "dimensions": dimension,
                "metric": metric,
                "n_queries": int(n_queries),
                "contrast_mean": float(contrast.mean()),
                "contrast_std": float(contrast.std(ddof=1)) if n_queries > 1 else 0.0,
                "d_min_mean": float(d_min.mean()),
                "d_max_mean": float(d_max.mean()),
            }
        )

    return pd.DataFrame(rows)


def run_task2(
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_queries: int = 20,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Measure distance concentration for every (sample size, metric) combination."""
    frames = [
        compute_contrast(
            sample_size=sample_size,
            dimensions=dimensions,
            metric=metric,
            n_queries=n_queries,
            seed=seed,
        )
        for metric in metrics
        for sample_size in sample_sizes
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Task 3: feature selection with Random Forest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Task3Output:
    """Results of the feature-selection experiment."""

    results: pd.DataFrame
    random_baselines: pd.DataFrame
    feature_importance: pd.DataFrame


def evaluate_subset(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    seed: int = DEFAULT_SEED,
    return_model: bool = False,
):
    """Train a Random Forest and return accuracy, macro-F1, time and memory."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_ms = (time.perf_counter() - start) * 1000.0

    predictions = model.predict(X_test)
    metrics = {
        "accuracy_pct": accuracy_score(y_test, predictions) * 100.0,
        "macro_f1_pct": f1_score(y_test, predictions, average="macro") * 100.0,
        "train_time_ms": float(train_time_ms),
        "memory_kb": (X_train.nbytes + X_test.nbytes) / 1024.0,
    }
    return (metrics, model) if return_model else metrics


def run_task3(
    variance_thresholds: Sequence[float] = DEFAULT_VARIANCE_THRESHOLDS,
    correlation_threshold: float = 0.9,
    random_trials: int = 10,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    n_estimators: int = 100,
) -> Task3Output:
    """Compare variance filtering, correlation filtering and random subsets.

    Random baselines are matched to each reduced feature count so that a
    selection method is compared against dropping the same number of features
    at random, not against the full baseline alone.
    """
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, seed=seed, stratify=True
    )
    total_features = int(X_train.shape[1])

    baseline, baseline_model = evaluate_subset(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=n_estimators,
        seed=seed,
        return_model=True,
    )
    rows: list[dict] = [
        {
            "method": "Baseline (all features)",
            "family": "Baseline",
            "detail": "No selection",
            "n_features": total_features,
            **baseline,
        }
    ]

    # --- Variance threshold sweep -----------------------------------------
    for threshold in sorted({float(t) for t in variance_thresholds}):
        selector = VarianceThreshold(threshold=threshold)
        X_train_var = selector.fit_transform(X_train)
        X_test_var = selector.transform(X_test)
        n_features = int(X_train_var.shape[1])

        if n_features == 0:
            continue

        rows.append(
            {
                "method": f"Variance threshold = {threshold:g}",
                "family": "Variance",
                "detail": f"Variance > {threshold:g}",
                "n_features": n_features,
                **evaluate_subset(
                    X_train_var,
                    X_test_var,
                    y_train,
                    y_test,
                    n_estimators=n_estimators,
                    seed=seed,
                ),
            }
        )

    # --- Correlation filter ------------------------------------------------
    correlation = pd.DataFrame(X_train).corr().abs().fillna(0.0)
    upper_triangle = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    to_drop = {
        int(column)
        for column in upper_triangle.columns
        if (upper_triangle[column] > correlation_threshold).any()
    }
    keep = [column for column in range(total_features) if column not in to_drop]

    if keep:
        X_train_corr = X_train[:, keep]
        X_test_corr = X_test[:, keep]
        rows.append(
            {
                "method": f"Correlation filter (|r| > {correlation_threshold:g})",
                "family": "Correlation",
                "detail": f"Dropped {len(to_drop)} correlated features",
                "n_features": int(X_train_corr.shape[1]),
                **evaluate_subset(
                    X_train_corr,
                    X_test_corr,
                    y_train,
                    y_test,
                    n_estimators=n_estimators,
                    seed=seed,
                ),
            }
        )

    results = pd.DataFrame(rows)

    # --- Matched random baselines -----------------------------------------
    reduced_sizes = sorted(
        {
            int(size)
            for size in results.loc[results["family"] != "Baseline", "n_features"]
            if 0 < int(size) < total_features
        }
    )
    rng = np.random.default_rng(seed)
    random_rows = []

    for size in reduced_sizes:
        accuracies, f1_scores, times, memories = [], [], [], []
        for trial in range(max(1, int(random_trials))):
            columns = rng.choice(total_features, size=size, replace=False)
            metrics = evaluate_subset(
                X_train[:, columns],
                X_test[:, columns],
                y_train,
                y_test,
                n_estimators=n_estimators,
                seed=seed + trial,
            )
            accuracies.append(metrics["accuracy_pct"])
            f1_scores.append(metrics["macro_f1_pct"])
            times.append(metrics["train_time_ms"])
            memories.append(metrics["memory_kb"])

        random_rows.append(
            {
                "n_features": size,
                "trials": max(1, int(random_trials)),
                "accuracy_pct": float(np.mean(accuracies)),
                "accuracy_std_pct": float(np.std(accuracies, ddof=1))
                if len(accuracies) > 1
                else 0.0,
                "macro_f1_pct": float(np.mean(f1_scores)),
                "train_time_ms": float(np.mean(times)),
                "memory_kb": float(np.mean(memories)),
            }
        )

    importance = pd.DataFrame(
        {
            "feature": np.arange(total_features),
            "importance": baseline_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return Task3Output(
        results=results,
        random_baselines=pd.DataFrame(random_rows),
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(frame.to_string(index=False))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the AIT403 Tutorial 1 analyses from the command line."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        help="Tasks to run (default: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=max(DEFAULT_DIMENSIONS),
        help="Largest dimension used in Task 2 (lower it for a faster run).",
    )
    parser.add_argument("--random-trials", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if "1" in args.tasks:
        table = run_task1(test_size=args.test_size, seed=args.seed)
        _print_table("Task 1 - KNN with injected noise features", table.round(4))

    if "2" in args.tasks:
        dimensions = [d for d in DEFAULT_DIMENSIONS if d <= args.max_dimension]
        table = run_task2(dimensions=dimensions, seed=args.seed)
        _print_table("Task 2 - Distance concentration", table.round(4))

    if "3" in args.tasks:
        output = run_task3(
            random_trials=args.random_trials,
            test_size=args.test_size,
            seed=args.seed,
        )
        _print_table("Task 3 - Feature selection", output.results.round(4))
        if not output.random_baselines.empty:
            _print_table(
                "Task 3 - Matched random baselines",
                output.random_baselines.round(4),
            )
        _print_table(
            "Task 3 - Top 10 features (baseline Random Forest)",
            output.feature_importance.head(10).round(4),
        )


if __name__ == "__main__":
    main()
