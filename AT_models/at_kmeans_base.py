"""K-Means clustering experiment using the shared preprocessing pipeline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import (  # noqa: E402
    build_model_pipeline,
    fit_model,
    prepare_splits,
)
from src.preprocessing.pipeline import CLUSTERING_FEATURES  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "kmeans_pca.png"


def main():
    # k = 6 produces a slightly better silhouette score, but the smallest group is very tiny.
    # k = 5 has a pretty good silhouette score, while not having an unreasonably small group.
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    # No test set is needed for unsupervised learning. The clustering
    # preprocessor selects the configured feature list and scales it.
    train = prepare_splits("clustering")[0]
    X_train = train[0]

    pipeline = build_model_pipeline(
        "clustering",
        KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        ),
    )
    fit_model(pipeline, X_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_processed = preprocessor.transform(X_train)
    clusters = model.labels_

    print(f"K-Means clustering (k={k})")
    print("\nCluster sizes:")
    print(pd.Series(clusters).value_counts().sort_index())

    # Asks "Are observations closer to their own cluster than to neighboring clusters?"
    # Higher is better
    silhouette = silhouette_score(
        X_processed,
        clusters,
        sample_size=3000,
        random_state=42,
    )
    print(f"\nSilhouette score: {silhouette:.4f}")

    # "How much do clusters spread out relative to how separated their centers are?"
    # Lower is better
    davies_bouldin = davies_bouldin_score(
        X_processed,
        clusters,
    )
    print(f"Davies-Bouldin index: {davies_bouldin:.4f}")

    # Inspect clusters using the original feature values
    features = CLUSTERING_FEATURES
    results = X_train[features].copy()
    results["Cluster"] = clusters

    print("\nAverage feature values per cluster:")
    print(results.groupby("Cluster")[features].mean().round(2))

    # Booking segment profiling
    profile = X_train[
        ["ADR", "IsCanceled", "MarketSegment"]
    ].copy()
    profile["Cluster"] = clusters
    profile["MarketSegment"] = profile["MarketSegment"].fillna("Unknown")

    print("\nBusiness profile by cluster:")
    print(
        profile.groupby("Cluster").agg(
            Bookings=("ADR", "size"),
            MeanADR=("ADR", "mean"),
            MedianADR=("ADR", "median"),
            CancellationRate=("IsCanceled", "mean"),
        ).round(3)
    )

    print("\nMarket segment distribution (% within each cluster):")
    print(
        pd.crosstab(
            profile["Cluster"],
            profile["MarketSegment"],
            normalize="index",
        ).mul(100).round(1)
    )

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_processed)

    print("\nPCA explained variance:")
    print(pca.explained_variance_ratio_)
    print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")

    # Sample points to avoid plotting every booking
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_2d), size=min(5000, len(X_2d)), replace=False)

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        X_2d[indices, 0],
        X_2d[indices, 1],
        c=clusters[indices],
        cmap="tab10",
        s=8,
        alpha=0.4,
    )

    # Project the K-Means centroids into the same PCA space
    centroids_2d = pca.transform(model.cluster_centers_)

    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        color="black",
        marker="x",
        s=150,
        linewidths=3,
        label="Centroids",
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"K-Means Clusters (k={k}) - PCA Projection")
    plt.colorbar(scatter, ticks=range(k), label="Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
