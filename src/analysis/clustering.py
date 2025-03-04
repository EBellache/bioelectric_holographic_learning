# clustering.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def reduce_dimensionality(data_matrix, n_components=2):
    """Explicit PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_matrix)
    return reduced_data, pca


def cluster_states(data_matrix, n_clusters=2):
    """Explicit clustering of states using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_matrix)
    return cluster_labels, kmeans
