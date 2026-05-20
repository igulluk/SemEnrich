"""HDBSCAN clustering implementation."""

import numpy as np
from typing import Optional

from .base import BaseClusterer

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN (Hierarchical DBSCAN) clustering algorithm.
    
    Automatically determines the number of clusters based on density.
    More robust than DBSCAN as it builds a hierarchy of clusters.
    Noise points are labeled as -1.
    
    Attributes:
        min_cluster_size: Minimum size of clusters
        min_samples: Number of samples in a neighborhood for a point to be
                    considered a core point
        metric: Distance metric to use
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        random_seed: int = 42,
    ):
        """Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Number of samples in a neighborhood
            metric: Distance metric to use
            cluster_selection_epsilon: Distance threshold for cluster extraction
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed=random_seed)
        
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan is not installed. Please install it with: pip install hdbscan"
            )
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.model = None
        self.probabilities_ = None

    def fit(self, embeddings: np.ndarray) -> "HDBSCANClusterer":
        """Fit HDBSCAN to the embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            self: The fitted clusterer
        """
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        
        self.labels_ = self.model.fit_predict(embeddings)
        self.probabilities_ = self.model.probabilities_
        
        # Number of clusters (excluding noise)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.is_fitted = True
        
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings.
        
        Uses approximate prediction based on the trained model.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            labels: Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before prediction")
        
        # HDBSCAN has approximate_predict for new data
        labels, strengths = hdbscan.approximate_predict(self.model, embeddings)
        
        return labels

    def get_noise_count(self) -> int:
        """Get number of noise points (labeled as -1).
        
        Returns:
            count: Number of noise points
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return int(np.sum(self.labels_ == -1))

    def get_probabilities(self) -> np.ndarray:
        """Get cluster membership probabilities.
        
        Returns:
            probabilities: Array of membership strengths for each sample
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return self.probabilities_

    def get_cluster_statistics(self) -> dict:
        """Get statistics about the clustering.
        
        Returns:
            Dictionary with clustering statistics
        """
        base_stats = super().get_cluster_statistics()
        
        # Only compute statistics for non-noise points
        non_noise_probs = self.probabilities_[self.labels_ != -1]
        
        base_stats.update({
            "n_noise_points": self.get_noise_count(),
            "noise_ratio": float(self.get_noise_count() / len(self.labels_)),
            "mean_probability": float(np.mean(non_noise_probs)) if len(non_noise_probs) > 0 else 0.0,
            "min_probability": float(np.min(non_noise_probs)) if len(non_noise_probs) > 0 else 0.0,
        })
        
        return base_stats

    def get_condensed_tree(self):
        """Get the condensed tree for visualization.
        
        Returns:
            condensed_tree: The condensed cluster hierarchy
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return self.model.condensed_tree_

    def plot_condensed_tree(self):
        """Plot the condensed cluster hierarchy tree.
        
        Returns:
            plot: The tree plot
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        return self.model.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=('r', 'g', 'b', 'y', 'c', 'm')
        )

