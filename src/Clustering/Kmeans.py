import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

def train_parameterized_KMeans(name, train_set, n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=42):
    train_filename = f"{train_set}.csv"
    model_filename =  f"{name}_kmeans.joblib"
    
    train_set = pd.read_csv(data_dir / train_filename)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    kmeans.fit(train_set)

    show_results(kmeans, train_set)

    dump(kmeans, artifact_dir / model_filename)

def train_base_KMeans(name, train_set):
    train_filename = f"{train_set}.csv"
    model_filename =  f"{name}_kmeans.joblib"
    
    train_set = pd.read_csv(data_dir / train_filename)

    kmeans = KMeans()

    kmeans.fit(train_set)

    show_results(kmeans, train_set)

    dump(kmeans, artifact_dir / model_filename)

def predict_with_KMeans(name, pred_set):
    pred_filename = f"{pred_set}.csv"
    model_filename =  f"{name}_kmeans.joblib"

    pred_set = pd.read_csv(data_dir / pred_filename)
    kmeans = load(artifact_dir / model_filename)

    y_pred = kmeans.predict(pred_set)

    print(y_pred)

def show_results(kmeans, train_set):
    if hasattr(train_set, "values"):
        train_set = train_set.values

    sil = silhouette_score(train_set, kmeans.labels_)      
    dbi = davies_bouldin_score(train_set, kmeans.labels_)  

    print(f"Silhouette: {sil:.3f}")
    print(f"Davies-Bouldin: {dbi:.3f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(train_set)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(9, 6))
    
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('viridis', len(unique_labels))
    
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], 
        c=labels, 
        cmap=cmap, 
        alpha=0.6, 
        edgecolors='w', 
        linewidths=0.5
    )

    ax.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1], 
        marker='X', 
        s=250, 
        c='red', 
        edgecolors='black',
        label='Centroids'
    )

    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(i)), markersize=8, label=f'Cluster {i}')
        for i in unique_labels
    ]
    centroid_handle = plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Centroids')
    legend_handles.append(centroid_handle)

    ax.set_title('KMeans Clusters Visualized via PCA')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)')
    
    ax.legend(handles=legend_handles, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()