"""Base class for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""

    def __init__(self, random_seed: int = 42):
        """Initialize the clusterer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.labels_ = None
        self.n_clusters_ = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> "BaseClusterer":
        """Fit the clustering algorithm to the data.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            self: The fitted clusterer
        """
        pass

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            labels: Array of cluster labels
        """
        pass

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the algorithm and return cluster labels.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            labels: Array of cluster labels
        """
        self.fit(embeddings)
        return self.labels_

    def get_cluster_mapping(
        self, texts: List[str], embeddings: np.ndarray
    ) -> Dict[str, int]:
        """Create mapping from texts to cluster IDs.
        
        Args:
            texts: List of text strings
            embeddings: Corresponding embeddings
            
        Returns:
            Dictionary mapping text to cluster ID
        """
        if not self.is_fitted:
            self.fit(embeddings)
        
        return {text: int(label) for text, label in zip(texts, self.labels_)}

    def get_cluster_to_texts(
        self, texts: List[str], embeddings: np.ndarray
    ) -> Dict[int, List[str]]:
        """Create mapping from cluster IDs to texts.
        
        Args:
            texts: List of text strings
            embeddings: Corresponding embeddings
            
        Returns:
            Dictionary mapping cluster ID to list of texts
        """
        if not self.is_fitted:
            self.fit(embeddings)
        
        cluster_to_texts = {}
        for text, label in zip(texts, self.labels_):
            label = int(label)
            if label not in cluster_to_texts:
                cluster_to_texts[label] = []
            cluster_to_texts[label].append(text)
        
        return cluster_to_texts

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get the size of each cluster.
        
        Returns:
            Dictionary mapping cluster ID to size
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}

    def get_cluster_statistics(self) -> Dict[str, float]:
        """Get statistics about the clustering.
        
        Returns:
            Dictionary with clustering statistics
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted first")
        
        sizes = list(self.get_cluster_sizes().values())
        
        return {
            "n_clusters": int(self.n_clusters_),
            "mean_cluster_size": float(np.mean(sizes)),
            "std_cluster_size": float(np.std(sizes)),
            "min_cluster_size": int(np.min(sizes)),
            "max_cluster_size": int(np.max(sizes)),
            "median_cluster_size": float(np.median(sizes)),
        }

