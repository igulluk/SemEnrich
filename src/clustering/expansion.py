"""Graph-based expansion algorithm for findings augmentation."""

import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import itertools


class FindingsExpander:
    """Expand findings using graph-based compatible cluster addition."""

    def __init__(
        self,
        adjacency_threshold_norm: float = 0.001,
        adjacency_threshold_count: int = 3,
        only_positive_expansion: bool = True,
    ):
        """Initialize expander.
        
        Args:
            adjacency_threshold_norm: Normalized adjacency threshold
            adjacency_threshold_count: Minimum co-occurrence count
            only_positive_expansion: Only add positive (normal) clusters
        """
        self.adjacency_threshold_norm = adjacency_threshold_norm
        self.adjacency_threshold_count = adjacency_threshold_count
        self.only_positive_expansion = only_positive_expansion
        self.adjacency_matrix = None
        self.is_addable = None

    def build_adjacency_matrix(
        self,
        cluster_graphs: List[Tuple[int, ...]],
    ) -> np.ndarray:
        """Build adjacency matrix from cluster graphs.
        
        Args:
            cluster_graphs: List of cluster ID tuples representing findings
            
        Returns:
            Adjacency matrix of shape (n_clusters, n_clusters)
        """
        # Count graphs
        graph_counts = Counter(cluster_graphs)
        # Find maximum cluster ID
        max_cluster = 0
        for graph in cluster_graphs:
            if graph:
                max_cluster = max(max_cluster, max(graph))
        
        n_clusters = max_cluster + 1
        # Initialize adjacency matrix
        adj = np.zeros((n_clusters, n_clusters), dtype=int)
        
        # Fill adjacency matrix
        for graph, count in graph_counts.items():
            nodes = list(graph)
            for ni in nodes:
                for nj in nodes:
                    if ni != nj:
                        adj[ni, nj] += count
        
        self.adjacency_matrix = adj
        return adj

    def compute_addability_matrix(
        self,
        cluster_signs: Dict[int, str],
    ) -> np.ndarray:
        """Compute which clusters can be added to which.
        
        Args:
            cluster_signs: Dictionary mapping cluster ID to sign
            
        Returns:
            Binary matrix where is_addable[i,j]=1 means j can be added to a finding containing i
        """
        if self.adjacency_matrix is None:
            raise ValueError("Must build adjacency matrix first")
        
        n_clusters = self.adjacency_matrix.shape[0]
        
        # Normalize adjacency matrix
        row_sums = self.adjacency_matrix.sum(axis=1, keepdims=True)
        adj_normalized = self.adjacency_matrix / (row_sums + 1e-12)
        
        # Compute addability
        is_addable = np.zeros((n_clusters, n_clusters), dtype=int)
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                # Check thresholds
                if (adj_normalized[i, j] > self.adjacency_threshold_norm and
                    self.adjacency_matrix[i, j] > self.adjacency_threshold_count):
                    
                    # Check sign constraint
                    if self.only_positive_expansion:
                        # Only add positive clusters
                        if cluster_signs.get(j, "neutral") == "positive":
                            is_addable[i, j] = 1
                    else:
                        is_addable[i, j] = 1
        
        self.is_addable = is_addable
        return is_addable

    def get_compatible_nodes(
        self,
        current_nodes: Set[int],
    ) -> Set[int]:
        """Get nodes that are compatible with all current nodes.
        
        Args:
            current_nodes: Set of current node IDs
            
        Returns:
            Set of compatible node IDs
        """
        if self.is_addable is None:
            raise ValueError("Must compute addability matrix first")
        
        # Find nodes that can be added to all current nodes
        compatible = None
        
        for node in current_nodes:
            # Find nodes that can be added to this node
            addable_to_node = set(np.where(self.is_addable[node] == 1)[0])
            
            # Remove nodes already in the graph
            addable_to_node -= current_nodes
            
            # Intersect with compatible nodes so far
            if compatible is None:
                compatible = addable_to_node
            else:
                compatible &= addable_to_node
        
        return compatible if compatible else set()

    def find_maximal_expansions(
        self,
        original_graph: Tuple[int, ...],
    ) -> List[Tuple[int, ...]]:
        """Find all maximal expansions of a graph.
        
        Args:
            original_graph: Tuple of cluster IDs
            
        Returns:
            List of maximal expansion tuples
        """
        if self.is_addable is None:
            raise ValueError("Must compute addability matrix first")
        
        current_nodes = set(original_graph)
        
        # Find compatible nodes
        compatible = self.get_compatible_nodes(current_nodes)
        
        if not compatible:
            # No expansion possible
            return [original_graph]
        
        # Check which compatible nodes are compatible with each other
        # to find maximal sets
        expansions = []
        
        # Try all possible combinations of compatible nodes
        compatible_list = sorted(list(compatible))
        
        # Start with empty expansion
        self._find_maximal_recursive(
            current_nodes.copy(),
            compatible_list,
            0,
            expansions
        )
        
        # Remove duplicates and ensure they are truly maximal
        unique_expansions = []
        seen = set()
        
        for exp in expansions:
            exp_tuple = tuple(sorted(exp))
            if exp_tuple not in seen:
                seen.add(exp_tuple)
                unique_expansions.append(exp_tuple)
        
        # Filter to only maximal expansions
        maximal = []
        for i, exp1 in enumerate(unique_expansions):
            is_maximal = True
            for j, exp2 in enumerate(unique_expansions):
                if i != j and set(exp1).issubset(set(exp2)) and len(exp1) < len(exp2):
                    is_maximal = False
                    break
            if is_maximal:
                maximal.append(exp1)
        
        return maximal if maximal else [original_graph]

    def _find_maximal_recursive(
        self,
        current_nodes: Set[int],
        candidates: List[int],
        start_idx: int,
        results: List[Set[int]],
    ):
        """Recursively find maximal node sets.
        
        Args:
            current_nodes: Current set of nodes
            candidates: Candidate nodes to add
            start_idx: Starting index in candidates
            results: List to store results
        """
        # Try adding each candidate
        added_any = False
        
        for i in range(start_idx, len(candidates)):
            node = candidates[i]
            
            # Check if this node is compatible with all current nodes
            can_add = all(
                self.is_addable[existing, node] == 1
                for existing in current_nodes
            )
            
            if can_add:
                # Add this node and continue
                new_nodes = current_nodes.copy()
                new_nodes.add(node)
                self._find_maximal_recursive(new_nodes, candidates, i + 1, results)
                added_any = True
        
        # If we couldn't add any more nodes, this is a maximal set
        if not added_any:
            results.append(current_nodes.copy())

    def generate_expansions(
        self,
        training_data: List[Dict],
        thought_cluster: Dict[str, int],
        cluster_signs: Dict[int, str],
    ) -> Dict[Tuple[int, ...], Dict]:
        """Generate expansions for all training samples.
        
        Args:
            training_data: List of training samples with 'Findings' key
            thought_cluster: Dictionary mapping text to cluster ID
            cluster_signs: Dictionary mapping cluster ID to sign
            
        Returns:
            Dictionary mapping original graph to expansion info
        """
        print("Extracting cluster graphs from training data...")
        
        # Extract cluster graphs
        graphs = []
        for sample in training_data:
            findings = sample.get('Findings', '')
            
            # Split into sentences
            if findings.endswith("."):
                sentences = findings.split(".")[:-1]
            else:
                sentences = findings.split(".")
            
            sentences = [s.lower().strip() for s in sentences if s.strip()]
            
            # Map to cluster IDs
            cluster_ids = []
            for sentence in sentences:
                if sentence in thought_cluster:
                    cluster_ids.append(thought_cluster[sentence])
            
            if cluster_ids:
                graphs.append(tuple(sorted(cluster_ids)))
        
        # Count unique graphs
        graph_counts = Counter(graphs)
        print(f"Found {len(graph_counts)} unique graphs")
        
        # Build adjacency matrix
        print("Building adjacency matrix...")
        self.build_adjacency_matrix(graphs)
        
        # Compute addability
        print("Computing addability matrix...")
        self.compute_addability_matrix(cluster_signs)
        
        # Generate expansions
        print("Generating expansions...")
        expansion_results = {}
        
        unique_graphs = sorted(graph_counts.keys())
        for i, graph in enumerate(unique_graphs):
            if i % 100 == 0:
                print(f"Processing graph {i+1}/{len(unique_graphs)}...")
            
            if not graph:  # Skip empty graphs
                continue
            
            expansions = self.find_maximal_expansions(graph)
            
            expansion_results[graph] = {
                'count': graph_counts[graph],
                'expansions': expansions
            }
        
        print(f"Generated expansions for {len(expansion_results)} graphs")
        
        return expansion_results

    def get_expansion_statistics(
        self,
        expansion_results: Dict[Tuple[int, ...], Dict],
    ) -> Dict:
        """Get statistics about expansions.
        
        Args:
            expansion_results: Dictionary of expansion results
            
        Returns:
            Dictionary with statistics
        """
        total_graphs = len(expansion_results)
        total_samples = sum(r['count'] for r in expansion_results.values())
        
        expansion_sizes = []
        n_expansions_list = []
        
        for original, result in expansion_results.items():
            n_original = len(original)
            for expanded in result['expansions']:
                n_added = len(expanded) - n_original
                expansion_sizes.append(n_added)
            n_expansions_list.append(len(result['expansions']))
        
        return {
            'total_graphs': total_graphs,
            'total_samples': total_samples,
            'mean_expansion_size': float(np.mean(expansion_sizes)) if expansion_sizes else 0.0,
            'max_expansion_size': int(np.max(expansion_sizes)) if expansion_sizes else 0,
            'mean_n_expansions': float(np.mean(n_expansions_list)),
            'max_n_expansions': int(np.max(n_expansions_list)),
        }

