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

# Function to train a K-means Clustering model with the Scikit-learn custom parameters
def train_parameterized_KMeans(name, train_set, n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=42):
    # File Names
    train_filename = f"{train_set}.csv"
    model_filename =  f"{name}_kmeans.joblib"
    
    # Load Training Dataset (No test set is needed for unsupervised learning)
    train_set = pd.read_csv(data_dir / train_filename)

    # Initialize the KMeans model with parameters
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    # Train
    kmeans.fit(train_set)

    # Calculate clustering metrics and display them
    show_results(kmeans, train_set)

    # Save trained model as an artifact
    dump(kmeans, artifact_dir / model_filename)

# Function to train a K-means Clustering model with the Scikit-learn default parameters
def train_base_KMeans(name, train_set):
    # File Names
    train_filename = f"{train_set}.csv"
    model_filename =  f"{name}_kmeans.joblib"
    
    # Load Training Dataset (No test set is needed for unsupervised learning)
    train_set = pd.read_csv(data_dir / train_filename)

    # Initialize the KMeans model without parameters
    kmeans = KMeans()

    # Train
    kmeans.fit(train_set)

    # Calculate clustering metrics and display them
    show_results(kmeans, train_set)

    # Save trained model as an artifact
    dump(kmeans, artifact_dir / model_filename)

# Function to load an artifact and make predictions on a new dataset
def predict_with_KMeans(name, pred_set):
    # File Names
    pred_filename = f"{pred_set}.csv"
    model_filename =  f"{name}_kmeans.joblib"

    # Load the prediction dataset and the trained model
    pred_set = pd.read_csv(data_dir / pred_filename)
    kmeans = load(artifact_dir / model_filename)

    # Make predictions using the loaded model
    y_pred = kmeans.predict(pred_set)

    # Print results
    print(y_pred)

# Function to display the results of the model
def show_results(kmeans, train_set):
    if hasattr(train_set, "values"):
        train_set = train_set.values

    # Silhouette Score: A measure of how similar an object is to its own cluster compared to other clusters
    # Davies-Bouldin Index: A measure of the average similarity ratio of each cluster with its most similar cluster
    sil = silhouette_score(train_set, kmeans.labels_)      
    dbi = davies_bouldin_score(train_set, kmeans.labels_)  

    print(f"Silhouette: {sil:.3f}")
    print(f"Davies-Bouldin: {dbi:.3f}")

    # Perform PCA to reduce the dimensionality of the data to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(train_set)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Create a scatter plot with matplotlib to visualize the clusters and their centroids
    fig, ax = plt.subplots(figsize=(9, 6))
    
    labels = kmeans.labels_
    unique_labels = np.unique(labels)

    # maps each label to it's cluster color in the scatter plot. The 'viridis' colormap is used to assign colors to each cluster.
    cmap = plt.get_cmap('viridis', len(unique_labels))

    # The data points
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], 
        c=labels, 
        cmap=cmap, 
        alpha=0.6, 
        edgecolors='w', 
        linewidths=0.5
    )

    # The centroids of the clusters
    ax.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1], 
        marker='X', 
        s=250, 
        c='red', 
        edgecolors='black',
        label='Centroids'
    )

    # The legend of the scatter plot
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(i)), markersize=8, label=f'Cluster {i}')
        for i in unique_labels
    ]
    centroid_handle = plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Centroids')
    legend_handles.append(centroid_handle)

    # Creating and displaying the plot
    ax.set_title('KMeans Clusters Visualized via PCA')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)')
    
    ax.legend(handles=legend_handles, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()