"""Clustering evaluation metrics and tools."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)


class ClusterEvaluator:
    """Evaluate clustering quality using various metrics."""

    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate clustering using specified metrics.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            metrics: List of metric names to compute. If None, compute all.
                    Available: 'silhouette', 'davies_bouldin', 'calinski_harabasz'
            
        Returns:
            Dictionary mapping metric name to score
        """
        if metrics is None:
            metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        
        results = {}
        
        # Filter out noise points for metrics that don't handle them
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        
        # Need at least 2 clusters for these metrics
        n_unique_labels = len(np.unique(filtered_labels))
        if n_unique_labels < 2:
            print(f"Warning: Only {n_unique_labels} unique cluster(s) found. "
                  "Some metrics require at least 2 clusters.")
            return results
        
        if 'silhouette' in metrics:
            try:
                score = silhouette_score(filtered_embeddings, filtered_labels)
                results['silhouette_score'] = float(score)
            except Exception as e:
                print(f"Error computing silhouette score: {e}")
        
        if 'davies_bouldin' in metrics:
            try:
                score = davies_bouldin_score(filtered_embeddings, filtered_labels)
                results['davies_bouldin_score'] = float(score)
            except Exception as e:
                print(f"Error computing Davies-Bouldin score: {e}")
        
        if 'calinski_harabasz' in metrics:
            try:
                score = calinski_harabasz_score(filtered_embeddings, filtered_labels)
                results['calinski_harabasz_score'] = float(score)
            except Exception as e:
                print(f"Error computing Calinski-Harabasz score: {e}")
        
        self.metrics = results
        return results

    def compute_silhouette_samples(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute silhouette coefficient for each sample.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            silhouette_values: Array of silhouette coefficients
        """
        # Filter out noise points
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        
        if len(np.unique(filtered_labels)) < 2:
            return np.zeros(len(embeddings))
        
        silhouette_values = silhouette_samples(filtered_embeddings, filtered_labels)
        
        # Map back to original indices
        result = np.zeros(len(embeddings))
        result[mask] = silhouette_values
        
        return result

    def compute_cluster_cohesion(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[int, float]:
        """Compute average within-cluster distance (cohesion).
        
        Lower values indicate more cohesive clusters.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Dictionary mapping cluster ID to cohesion score
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        cohesion = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) < 2:
                cohesion[int(cluster_id)] = 0.0
                continue
            
            # Compute pairwise distances within cluster
            distances = euclidean_distances(cluster_embeddings)
            # Average of upper triangle (excluding diagonal)
            avg_distance = distances[np.triu_indices_from(distances, k=1)].mean()
            cohesion[int(cluster_id)] = float(avg_distance)
        
        return cohesion

    def compute_cluster_separation(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[Tuple[int, int], float]:
        """Compute between-cluster distances (separation).
        
        Higher values indicate better separation.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Dictionary mapping cluster pair to separation distance
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        unique_labels = [l for l in np.unique(labels) if l != -1]
        separation = {}
        
        # Compute cluster centroids
        centroids = {}
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)
        
        # Compute pairwise centroid distances
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                dist = np.linalg.norm(centroids[label1] - centroids[label2])
                separation[(int(label1), int(label2))] = float(dist)
        
        return separation

    def compute_cluster_density(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[int, float]:
        """Compute cluster density (samples per unit volume).
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Dictionary mapping cluster ID to density
        """
        from scipy.spatial import ConvexHull
        
        density = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            n_samples = len(cluster_embeddings)
            
            if n_samples < embeddings.shape[1] + 1:
                # Not enough points to compute convex hull in this dimension
                density[int(cluster_id)] = float(n_samples)
                continue
            
            try:
                hull = ConvexHull(cluster_embeddings)
                volume = hull.volume
                density[int(cluster_id)] = float(n_samples / volume) if volume > 0 else float(n_samples)
            except:
                # Fallback: use variance as proxy for volume
                variance = np.var(cluster_embeddings, axis=0).sum()
                density[int(cluster_id)] = float(n_samples / variance) if variance > 0 else float(n_samples)
        
        return density

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics of computed metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        return self.metrics.copy()

    def compare_clusterings(
        self,
        embeddings: np.ndarray,
        labels_list: List[np.ndarray],
        names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple clustering results.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels_list: List of cluster label arrays
            names: List of names for each clustering
            
        Returns:
            Dictionary mapping clustering name to metrics
        """
        results = {}
        for labels, name in zip(labels_list, names):
            results[name] = self.evaluate(embeddings, labels)
        
        return results

    def print_report(self, embeddings: np.ndarray, labels: np.ndarray):
        """Print a comprehensive evaluation report.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
        """
        print("=" * 70)
        print("CLUSTERING EVALUATION REPORT")
        print("=" * 70)
        
        # Basic statistics
        n_samples = len(labels)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        
        print(f"\nBasic Statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise} ({100*n_noise/n_samples:.2f}%)")
        
        # Cluster sizes
        print(f"\nCluster Size Statistics:")
        sizes = [np.sum(labels == l) for l in unique_labels if l != -1]
        if sizes:
            print(f"  Mean: {np.mean(sizes):.1f}")
            print(f"  Std: {np.std(sizes):.1f}")
            print(f"  Min: {np.min(sizes)}")
            print(f"  Max: {np.max(sizes)}")
            print(f"  Median: {np.median(sizes):.1f}")
        
        # Quality metrics
        print(f"\nQuality Metrics:")
        metrics = self.evaluate(embeddings, labels)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        
        # Cohesion
        print(f"\nCluster Cohesion (avg within-cluster distance):")
        cohesion = self.compute_cluster_cohesion(embeddings, labels)
        if cohesion:
            cohesion_values = list(cohesion.values())
            print(f"  Mean: {np.mean(cohesion_values):.4f}")
            print(f"  Std: {np.std(cohesion_values):.4f}")
            print(f"  Min: {np.min(cohesion_values):.4f}")
            print(f"  Max: {np.max(cohesion_values):.4f}")
        
        print("=" * 70)

