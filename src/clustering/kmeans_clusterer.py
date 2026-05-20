"""K-Means clustering implementation."""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional

from .base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """K-Means clustering algorithm.
    
    Attributes:
        n_clusters: Number of clusters to form
        max_iter: Maximum number of iterations
        n_init: Number of time the k-means algorithm will be run
        tol: Relative tolerance for convergence
    """

    def __init__(
        self,
        n_clusters: int = 1000,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_seed: int = 42,
    ):
        """Initialize K-Means clusterer.
        
        Args:
            n_clusters: Number of clusters to form
            max_iter: Maximum number of iterations
            n_init: Number of time the k-means algorithm will be run
            tol: Relative tolerance for convergence
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed=random_seed)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.model = None
        self.cluster_centers_ = None

    def fit(self, embeddings: np.ndarray) -> "KMeansClusterer":
        """Fit K-Means to the embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            self: The fitted clusterer
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=self.random_seed,
            verbose=0,
        )
        
        self.labels_ = self.model.fit_predict(embeddings)
        self.cluster_centers_ = self.model.cluster_centers_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.is_fitted = True
        
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            labels: Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before prediction")
        
        return self.model.predict(embeddings)

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroids.
        
        Returns:
            Array of shape (n_clusters, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return self.cluster_centers_

    def get_distances_to_centers(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute distances from samples to their assigned cluster centers.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            distances: Array of distances
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        labels = self.predict(embeddings)
        distances = np.zeros(len(embeddings))
        
        for i, (embedding, label) in enumerate(zip(embeddings, labels)):
            center = self.cluster_centers_[label]
            distances[i] = np.linalg.norm(embedding - center)
        
        return distances

    def get_inertia(self) -> float:
        """Get sum of squared distances to nearest cluster center.
        
        Returns:
            inertia: Sum of squared distances
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return float(self.model.inertia_)

