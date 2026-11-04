
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score


# import sys # Used with sweep script.

# k = 6 produces a slightly better silhouette score, but the smallest group is very tiny.
# k = 5 has a pretty good silhouette score, while not having an unresonably small group.
# k = int(sys.argv[1]) # Used with sweep script.
k = 5

df = pd.read_csv("data/raw/H2.csv")

features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
]

X = df[features]

# Keep the same development partition as the other experiments
development_indices, test_indices = train_test_split(
    df.index,
    test_size=0.30,
    random_state=42,
)

X_development = X.loc[development_indices]

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_processed = preprocessor.fit_transform(X_development)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10,
)

clusters = model.fit_predict(X_processed)

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
results = X_development.copy()
results["Cluster"] = clusters

print("\nAverage feature values per cluster:")
print(results.groupby("Cluster")[features].mean().round(2))


# Booking segment profiling
profile = df.loc[
    X_development.index,
    ["ADR", "IsCanceled", "MarketSegment"]
].copy()

profile["Cluster"] = clusters
profile["MarketSegment"] = profile["MarketSegment"].str.strip()

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

# Sample points to avoid plotting all 55,531 bookings
rng = np.random.default_rng(42)
indices = rng.choice(len(X_2d), size=5000, replace=False)

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
plt.title(f"K-Means Clusters (k={k}) — PCA Projection")
plt.colorbar(scatter, ticks=range(k), label="Cluster")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_pca.png", dpi=200, bbox_inches="tight")
plt.show()