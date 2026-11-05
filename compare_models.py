"""Compare every merged model on the same train/validation/test protocol.

Every model uses the preprocessing pipeline from `src/preprocessing/`, fitted
on the training split only. Results are written to `outputs/` as a CSV, a
Markdown report and a summary figure.

Usage:
    python compare_models.py
    python compare_models.py --skip-knn
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    LinearRegression,
    LogisticRegression,
    SGDRegressor,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import build_model_pipeline, prepare_splits  # noqa: E402
from src.preprocessing.pipeline import JUDGMENT_COLUMNS  # noqa: E402

METRIC_ORDER = {
    "classification": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ],
    "regression": ["mae", "rmse", "r2"],
    "clustering": [
        "silhouette",
        "davies_bouldin",
        "min_cluster_size",
        "inertia",
    ],
}

# ============================================================================
# Model definitions
# ============================================================================

def classification_models(skip_knn=False):
    models = [
        (
            "Dummy (majority)",
            DummyClassifier(strategy="most_frequent"),
            (),
        ),
        (
            "Logistic Regression (base)",
            LogisticRegression(max_iter=1000, random_state=42),
            (),
        ),
        (
            "Logistic Regression (extended)",
            LogisticRegression(max_iter=1000, random_state=42),
            tuple(JUDGMENT_COLUMNS["classification"]),
        ),
        (
            "KNN (k=5)",
            KNeighborsClassifier(n_jobs=-1),
            (),
        ),
    ]
    if skip_knn:
        models = [m for m in models if not m[0].startswith("KNN")]
    return models


def regression_models():
    return [
        (
            "Dummy (mean)",
            DummyRegressor(strategy="mean"),
            (),
        ),
        (
            "Linear Regression (base)",
            LinearRegression(),
            (),
        ),
        (
            "Linear Regression (extended)",
            LinearRegression(),
            tuple(JUDGMENT_COLUMNS["regression"]),
        ),
        (
            "SGD Regressor",
            SGDRegressor(random_state=42),
            (),
        ),
    ]


def clustering_models():
    return [
        (
            f"KMeans (k={k})",
            KMeans(n_clusters=k, n_init=10, random_state=42),
            (),
        )
        for k in range(2, 9)
    ]


# ============================================================================
# Evaluation
# ============================================================================

def record(records, task, model, split, metric, value):
    records.append(
        {
            "task": task,
            "model": model,
            "split": split,
            "metric": metric,
            "value": np.nan if value is None else float(value),
        }
    )


def evaluate_classification(records, train_data, val_data, test_data, skip_knn):
    X_train, y_train, _ = train_data

    for name, estimator, include_judgment in classification_models(skip_knn):
        pipeline = build_model_pipeline(
            "classification",
            clone(estimator),
            include_judgment=include_judgment,
        )
        pipeline.fit(X_train, y_train)
        print(f"[classification] trained {name}")

        for split, (X, y, _) in (
            ("validation", val_data),
            ("test", test_data),
        ):
            predictions = pipeline.predict(X)
            probabilities = pipeline.predict_proba(X)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(
                    y, predictions, zero_division=0
                ),
                "recall": recall_score(y, predictions, zero_division=0),
                "f1": f1_score(y, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y, probabilities),
                "pr_auc": average_precision_score(y, probabilities),
            }

            for metric, value in metrics.items():
                record(records, "classification", name, split, metric, value)


def evaluate_regression(records, train_data, val_data, test_data):
    X_train, y_train, _ = train_data

    for name, estimator, include_judgment in regression_models():
        pipeline = build_model_pipeline(
            "regression",
            clone(estimator),
            include_judgment=include_judgment,
        )
        pipeline.fit(X_train, y_train)
        print(f"[regression] trained {name}")

        for split, (X, y, _) in (
            ("validation", val_data),
            ("test", test_data),
        ):
            predictions = pipeline.predict(X)
            mse = mean_squared_error(y, predictions)

            metrics = {
                "mae": mean_absolute_error(y, predictions),
                "rmse": np.sqrt(mse),
                "r2": r2_score(y, predictions),
            }

            for metric, value in metrics.items():
                record(records, "regression", name, split, metric, value)


def evaluate_clustering(records, train_data, val_data, test_data):
    X_train = train_data[0]

    for name, estimator, _ in clustering_models():
        pipeline = build_model_pipeline("clustering", clone(estimator))
        pipeline.fit(X_train)
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        print(f"[clustering] trained {name}")

        record(records, "clustering", name, "training", "inertia", model.inertia_)

        for split, (X, _, _) in (
            ("validation", val_data),
            ("test", test_data),
        ):
            labels = pipeline.predict(X)
            transformed = preprocessor.transform(X)
            n_labels = len(np.unique(labels))

            if n_labels > 1:
                silhouette = silhouette_score(
                    transformed,
                    labels,
                    sample_size=min(3000, len(transformed)),
                    random_state=42,
                )
                davies_bouldin = davies_bouldin_score(transformed, labels)
            else:
                silhouette = np.nan
                davies_bouldin = np.nan

            sizes = np.bincount(labels, minlength=model.n_clusters)

            metrics = {
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "min_cluster_size": int(sizes.min()),
            }

            for metric, value in metrics.items():
                record(records, "clustering", name, split, metric, value)


# ============================================================================
# Reporting
# ============================================================================

def ordered_models(df, task):
    return list(df.loc[df["task"] == task, "model"].drop_duplicates())


def pivot_table(df, task, split):
    subset = df[
        (df["task"] == task) & (df["split"] == split)
    ]
    if subset.empty:
        return None

    table = subset.pivot_table(
        index="model",
        columns="metric",
        values="value",
        aggfunc="first",
    )

    metrics = [
        metric
        for metric in METRIC_ORDER[task]
        if metric in table.columns
    ]
    table = table[metrics]

    order = [m for m in ordered_models(df, task) if m in table.index]
    return table.loc[order]


def markdown_table(table):
    headers = ["model"] + list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for model, row in table.iterrows():
        cells = [model]
        for column in table.columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif column == "min_cluster_size":
                cells.append(str(int(value)))
            else:
                cells.append(f"{value:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def write_report(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_comparison.csv", index=False)

    lines = [
        "# Model comparison",
        "",
        "All models are trained on the same training split and evaluated on",
        "the same validation/test splits created by `src.data.loader`.",
        "The preprocessing pipeline is fitted on the training split only.",
        "",
    ]

    for task in ("classification", "regression", "clustering"):
        lines.extend([f"## {task.title()}", ""])

        for split in ("validation", "test", "training"):
            table = pivot_table(df, task, split)
            if table is None:
                continue
            lines.extend(
                [
                    f"### {split.title()} split",
                    "",
                    markdown_table(table),
                    "",
                ]
            )

    report_path = output_dir / "model_comparison.md"
    report_path.write_text("\n".join(lines))
    return report_path


def plot_summary(df, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Classification: test F1 / ROC-AUC / PR-AUC
    classification = df[
        (df["task"] == "classification")
        & (df["split"] == "test")
        & (df["metric"].isin(["f1", "roc_auc", "pr_auc"]))
    ]
    if not classification.empty:
        table = classification.pivot(
            index="model", columns="metric", values="value"
        )
        table.plot(kind="bar", ax=axes[0, 0], rot=20)
    axes[0, 0].set_title("Classification - test")
    axes[0, 0].set_ylabel("score")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(loc="lower right")

    # Regression: test R2
    regression = df[
        (df["task"] == "regression")
        & (df["split"] == "test")
        & (df["metric"] == "r2")
    ]
    if not regression.empty:
        table = regression.pivot(
            index="model", columns="metric", values="value"
        )
        table.plot(kind="bar", ax=axes[0, 1], rot=20, color="tab:green")
    axes[0, 1].set_title("Regression - test R2")
    axes[0, 1].set_ylabel("R2")

    # Regression: test RMSE
    regression = df[
        (df["task"] == "regression")
        & (df["split"] == "test")
        & (df["metric"] == "rmse")
    ]
    if not regression.empty:
        table = regression.pivot(
            index="model", columns="metric", values="value"
        )
        table.plot(kind="bar", ax=axes[1, 0], rot=20, color="tab:orange")
    axes[1, 0].set_title("Regression - test RMSE")
    axes[1, 0].set_ylabel("RMSE (lower is better)")

    # Clustering: validation silhouette
    clustering = df[
        (df["task"] == "clustering")
        & (df["split"] == "validation")
        & (df["metric"] == "silhouette")
    ]
    if not clustering.empty:
        table = clustering.pivot(
            index="model", columns="metric", values="value"
        )
        table.plot(kind="bar", ax=axes[1, 1], rot=20, color="tab:purple")
    axes[1, 1].set_title("Clustering - validation silhouette")
    axes[1, 1].set_ylabel("silhouette (higher is better)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_tables(df):
    for task in ("classification", "regression", "clustering"):
        print(f"\n{'=' * 70}")
        print(task.upper())
        print("=" * 70)

        for split in ("validation", "test", "training"):
            table = pivot_table(df, task, split)
            if table is None:
                continue
            print(f"\n[{split}]")
            print(table.to_string(float_format=lambda v: f"{v:.4f}"))


# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare all models on the shared data splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory for the report and figure.",
    )
    parser.add_argument(
        "--skip-knn",
        action="store_true",
        help="Skip KNN (the slowest model).",
    )
    args = parser.parse_args()

    records = []

    print("Preparing classification data...")
    train, val, test = prepare_splits("classification")
    evaluate_classification(records, train, val, test, args.skip_knn)

    print("\nPreparing regression data...")
    train, val, test = prepare_splits("regression")
    evaluate_regression(records, train, val, test)

    print("\nPreparing clustering data...")
    train, val, test = prepare_splits("clustering")
    evaluate_clustering(records, train, val, test)

    df = pd.DataFrame(records)
    report_path = write_report(df, args.output_dir)
    plot_summary(df, args.output_dir / "model_comparison.png")
    print_tables(df)

    print(f"\nCSV:      {args.output_dir / 'model_comparison.csv'}")
    print(f"Markdown: {report_path}")
    print(f"Figure:   {args.output_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
