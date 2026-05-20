"""DBSCAN clustering implementation."""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import Optional

from .base import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """DBSCAN (Density-Based Spatial Clustering) algorithm.
    
    Automatically determines the number of clusters based on density.
    Noise points are labeled as -1.
    
    Attributes:
        eps: Maximum distance between two samples for one to be considered
             as in the neighborhood of the other
        min_samples: Number of samples in a neighborhood for a point to be
                    considered as a core point
        metric: Metric used for distance computation
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        random_seed: int = 42,
    ):
        """Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between two samples
            min_samples: Number of samples in a neighborhood
            metric: Distance metric to use
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed=random_seed)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = None
        self.core_sample_indices_ = None

    def fit(self, embeddings: np.ndarray) -> "DBSCANClusterer":
        """Fit DBSCAN to the embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            self: The fitted clusterer
        """
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        
        self.labels_ = self.model.fit_predict(embeddings)
        self.core_sample_indices_ = self.model.core_sample_indices_
        
        # Number of clusters (excluding noise)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.is_fitted = True
        
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings.
        
        Note: DBSCAN doesn't have a standard predict method.
        This implementation assigns new points to the nearest core point's cluster.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            labels: Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before prediction")
        
        # For DBSCAN, we need to store the training data
        if not hasattr(self, 'training_embeddings_'):
            raise ValueError(
                "Cannot predict with DBSCAN without storing training data. "
                "Consider using fit_predict instead."
            )
        
        # Assign each new point to the nearest core point's cluster
        from sklearn.metrics.pairwise import euclidean_distances
        
        core_embeddings = self.training_embeddings_[self.core_sample_indices_]
        core_labels = self.labels_[self.core_sample_indices_]
        
        distances = euclidean_distances(embeddings, core_embeddings)
        nearest_core = np.argmin(distances, axis=1)
        
        return core_labels[nearest_core]

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit DBSCAN and return cluster labels.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            labels: Array of cluster labels
        """
        # Store embeddings for later prediction
        self.training_embeddings_ = embeddings.copy()
        return super().fit_predict(embeddings)

    def get_noise_count(self) -> int:
        """Get number of noise points (labeled as -1).
        
        Returns:
            count: Number of noise points
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return int(np.sum(self.labels_ == -1))

    def get_core_sample_count(self) -> int:
        """Get number of core samples.
        
        Returns:
            count: Number of core samples
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return len(self.core_sample_indices_)

    def get_cluster_statistics(self) -> dict:
        """Get statistics about the clustering.
        
        Returns:
            Dictionary with clustering statistics
        """
        base_stats = super().get_cluster_statistics()
        
        base_stats.update({
            "n_noise_points": self.get_noise_count(),
            "n_core_samples": self.get_core_sample_count(),
            "noise_ratio": float(self.get_noise_count() / len(self.labels_)),
        })
        
        return base_stats

