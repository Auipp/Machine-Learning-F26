"""K-Means clustering using the shared preprocessing pipeline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import (  # noqa: E402
    build_model_pipeline,
    fit_model,
    load_model,
    prediction_features,
    prepare_splits,
)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

# Silhouette/Davies-Bouldin are quadratic in the number of rows, so the full
# dataset is sampled for evaluation while KMeans is still fitted on all rows.
MAX_EVAL_ROWS = 5000


# Train a K-Means Clustering model with custom parameters
def train_parameterized_KMeans(
    name,
    n_clusters=3,
    init="random",
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=42,
    duplicate_weight=0.0,
):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    return _train_kmeans(name, kmeans, duplicate_weight)


# Train a K-Means Clustering model with the Scikit-learn default parameters
def train_base_KMeans(name, duplicate_weight=0.0):
    return _train_kmeans(name, KMeans(), duplicate_weight)


def _train_kmeans(name, kmeans, duplicate_weight):
    # No test set is needed for unsupervised learning. The preprocessing
    # pipeline selects the clustering feature whitelist and scales it.
    train = prepare_splits("clustering", duplicate_weight)[0]
    X_train, _, w_train = train

    pipeline = build_model_pipeline("clustering", kmeans)
    fit_model(
        pipeline,
        X_train,
        sample_weight=w_train if duplicate_weight else None,
    )

    # Calculate clustering metrics and display them
    show_results(pipeline, X_train)

    # Save the preprocessing pipeline together with the trained model
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, artifact_dir / f"{name}_kmeans.joblib")
    return pipeline


# Load an artifact and make predictions on a new dataset
def predict_with_KMeans(name, pred_set="test"):
    """Predict cluster labels for a split name or a raw CSV path."""
    pipeline = load_model(artifact_dir / f"{name}_kmeans.joblib")
    features = prediction_features(pred_set, "clustering", data_dir)

    y_pred = pipeline.predict(features)
    print(y_pred)
    return y_pred


# Display the results of the model
def show_results(pipeline, X_train, max_eval_rows=MAX_EVAL_ROWS):
    preprocessor = pipeline.named_steps["preprocessor"]
    kmeans = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(X_train)
    labels = kmeans.labels_

    if len(X_processed) > max_eval_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_processed), size=max_eval_rows, replace=False)
        X_eval, labels_eval = X_processed[idx], labels[idx]
    else:
        X_eval, labels_eval = X_processed, labels

    # Silhouette Score: how similar an object is to its own cluster compared
    # to other clusters. Davies-Bouldin: average similarity of each cluster
    # with its most similar cluster (lower is better).
    sil = silhouette_score(X_eval, labels_eval)
    dbi = davies_bouldin_score(X_eval, labels_eval)

    print(f"Silhouette: {sil:.3f}")
    print(f"Davies-Bouldin: {dbi:.3f}")

    # Perform PCA to reduce the dimensionality of the data to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_eval)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Create a scatter plot with matplotlib to visualize the clusters and their centroids
    fig, ax = plt.subplots(figsize=(9, 6))

    unique_labels = np.unique(labels_eval)

    # maps each label to its cluster color in the scatter plot
    cmap = plt.get_cmap("viridis", len(unique_labels))

    # The data points
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels_eval,
        cmap=cmap,
        alpha=0.6,
        edgecolors="w",
        linewidths=0.5,
    )

    # The centroids of the clusters
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=250,
        c="red",
        edgecolors="black",
        label="Centroids",
    )

    # The legend of the scatter plot
    norm = plt.Normalize(vmin=labels_eval.min(), vmax=labels_eval.max())
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(norm(i)),
            markersize=8,
            label=f"Cluster {i}",
        )
        for i in unique_labels
    ]
    centroid_handle = plt.Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=10,
        label="Centroids",
    )
    legend_handles.append(centroid_handle)

    # Creating and displaying the plot
    ax.set_title("KMeans Clusters Visualized via PCA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)")

    ax.legend(handles=legend_handles, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    train_base_KMeans("A")
