
import pandas as pd

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