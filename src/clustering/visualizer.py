"""Visualization tools for clustering analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ClusterVisualizer:
    """Visualize clustering results."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_2d_projection(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Plot 2D projection of high-dimensional embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            title: Plot title
            save_path: Path to save the figure
        """
        # Reduce to 2D
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                color = 'black'
                marker = 'x'
                label_str = 'Noise'
            else:
                marker = 'o'
                label_str = f'Cluster {label}'
            
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=label_str,
                marker=marker,
                alpha=0.6,
                s=30,
            )
        
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Cluster Visualization using {method.upper()}')
        
        # Show legend only if not too many clusters
        if len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_cluster_sizes(
        self,
        labels: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Plot histogram of cluster sizes.
        
        Args:
            labels: Cluster labels for each sample
            title: Plot title
            save_path: Path to save the figure
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Remove noise if present
        if -1 in unique_labels:
            noise_idx = np.where(unique_labels == -1)[0][0]
            noise_count = counts[noise_idx]
            unique_labels = np.delete(unique_labels, noise_idx)
            counts = np.delete(counts, noise_idx)
        else:
            noise_count = 0
        
        plt.figure(figsize=self.figsize)
        
        plt.bar(range(len(counts)), sorted(counts, reverse=True))
        plt.xlabel('Cluster Index (sorted by size)')
        plt.ylabel('Number of Samples')
        
        if title:
            plt.title(title)
        else:
            plt.title('Cluster Size Distribution')
        
        # Add statistics
        stats_text = f'Mean: {counts.mean():.1f}\nStd: {counts.std():.1f}\n'
        stats_text += f'Min: {counts.min()}\nMax: {counts.max()}\n'
        if noise_count > 0:
            stats_text += f'Noise: {noise_count}'
        
        plt.text(
            0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_silhouette(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Plot silhouette analysis.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            title: Plot title
            save_path: Path to save the figure
        """
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        # Filter noise
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        
        if len(np.unique(filtered_labels)) < 2:
            print("Cannot compute silhouette with less than 2 clusters")
            return
        
        silhouette_avg = silhouette_score(filtered_embeddings, filtered_labels)
        sample_silhouette_values = silhouette_samples(filtered_embeddings, filtered_labels)
        
        plt.figure(figsize=self.figsize)
        
        y_lower = 10
        unique_labels = np.unique(filtered_labels)
        
        for i, label in enumerate(sorted(unique_labels)):
            cluster_silhouette_values = sample_silhouette_values[filtered_labels == label]
            cluster_silhouette_values.sort()
            
            size_cluster = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster
            
            color = plt.cm.nipy_spectral(float(i) / len(unique_labels))
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
            y_lower = y_upper + 10
        
        plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average: {silhouette_avg:.3f}')
        
        plt.xlabel('Silhouette Coefficient')
        plt.ylabel('Cluster')
        
        if title:
            plt.title(title)
        else:
            plt.title('Silhouette Analysis')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_distance_distribution(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_clusters_to_show: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Plot distribution of distances to cluster centroids.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            n_clusters_to_show: Number of largest clusters to show
            title: Plot title
            save_path: Path to save the figure
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Get largest clusters
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        largest_clusters = unique_labels[np.argsort(counts)[-n_clusters_to_show:]]
        
        plt.figure(figsize=self.figsize)
        
        for cluster_id in largest_clusters:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            
            # Compute distances
            distances = euclidean_distances(cluster_embeddings, centroid).flatten()
            
            plt.hist(distances, bins=30, alpha=0.5, label=f'Cluster {cluster_id}')
        
        plt.xlabel('Distance to Centroid')
        plt.ylabel('Frequency')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Distance Distribution (Top {n_clusters_to_show} Clusters)')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_adjacency_matrix(
        self,
        adjacency: np.ndarray,
        max_clusters: int = 50,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Plot adjacency matrix heatmap.
        
        Args:
            adjacency: Adjacency matrix of cluster co-occurrences
            max_clusters: Maximum number of clusters to show
            title: Plot title
            save_path: Path to save the figure
        """
        # Limit size for visualization
        if adjacency.shape[0] > max_clusters:
            adjacency = adjacency[:max_clusters, :max_clusters]
        
        plt.figure(figsize=self.figsize)
        
        sns.heatmap(
            adjacency,
            cmap='viridis',
            xticklabels=10,
            yticklabels=10,
            cbar_kws={'label': 'Co-occurrence Count'}
        )
        
        plt.xlabel('Cluster ID')
        plt.ylabel('Cluster ID')
        
        if title:
            plt.title(title)
        else:
            plt.title('Cluster Co-occurrence Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Plot comparison of metrics across different clustering methods.
        
        Args:
            results: Dictionary mapping method name to metrics dictionary
            title: Plot title
            save_path: Path to save the figure
        """
        # Prepare data
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [results[method].get(metric, 0) for method in methods]
            
            axes[i].bar(range(len(methods)), values)
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels(methods, rotation=45, ha='right')
            axes[i].set_ylabel('Score')
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].grid(axis='y', alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)
        else:
            fig.suptitle('Clustering Method Comparison', fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def create_report(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_dir: str,
        prefix: str = "clustering",
    ):
        """Create a comprehensive visualization report.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Cluster labels for each sample
            output_dir: Directory to save plots
            prefix: Prefix for plot filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating visualization report...")
        
        # 2D projections
        for method in ['tsne', 'pca']:
            self.plot_2d_projection(
                embeddings, labels, method=method,
                save_path=str(output_path / f"{prefix}_2d_{method}.png")
            )
        
        # Cluster sizes
        self.plot_cluster_sizes(
            labels,
            save_path=str(output_path / f"{prefix}_sizes.png")
        )
        
        # Silhouette
        self.plot_silhouette(
            embeddings, labels,
            save_path=str(output_path / f"{prefix}_silhouette.png")
        )
        
        # Distance distribution
        self.plot_distance_distribution(
            embeddings, labels,
            save_path=str(output_path / f"{prefix}_distances.png")
        )
        
        print(f"Report saved to {output_dir}")

