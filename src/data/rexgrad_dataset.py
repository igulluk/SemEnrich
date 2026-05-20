"""RexGrad dataset with support for clustering-based expansion."""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_dataset import BaseDataset
from ..utils.io_utils import load_json
from ..utils.text_utils import split_sentences, preprocess_text


class RexGradDataset(BaseDataset):
    """RexGrad medical report dataset with flexible configuration."""

    def __init__(
        self,
        data_path: str,
        phase: str = "train",
        n_samples: Optional[int] = None,
        use_findings_only: bool = True,
        use_expansion: bool = False,
        cluster_file: Optional[str] = None,
        expansion_file: Optional[str] = None,
        use_random_from_cluster: bool = False,
        transform: Optional[Any] = None,
        random_seed: int = 42,
        expansion_ablation: bool = False,
    ):
        """Initialize RexGrad dataset.
        
        Args:
            data_path: Path to data directory
            phase: Dataset phase ('train', 'valid', 'test')
            n_samples: Number of samples to use (for subsets), None for all
            use_findings_only: Use only findings, not impression
            use_expansion: Use cluster-based expansion
            cluster_file: Path to cluster mapping file (JSON)
            expansion_file: Path to expansion results file (JSON)
            use_random_from_cluster: Replace sentences with random ones from same cluster
            transform: Image transformations
            random_seed: Random seed
        """
        self.n_samples = n_samples
        self.use_findings_only = use_findings_only
        self.use_expansion = use_expansion
        self.use_random_from_cluster = use_random_from_cluster
        self.expansion_ablation = expansion_ablation
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load clustering data if needed
        self.thought_cluster = None
        self.cluster_to_texts = None
        self.expansion_graphs = None
        
        if use_expansion or use_random_from_cluster:
            if cluster_file is None:
                raise ValueError("cluster_file required when using expansion or use_random_from_cluster")
            
            self.thought_cluster = load_json(cluster_file)
            
            # Build cluster_to_texts mapping
            from collections import defaultdict
            self.cluster_to_texts = defaultdict(list)
            for text, cluster in self.thought_cluster.items():
                self.cluster_to_texts[cluster].append(text)
            
            if use_expansion:
                if expansion_file is None:
                    raise ValueError("expansion_file required when using expansion")
                self.expansion_graphs = self._load_expansion_results(expansion_file)
        
        super().__init__(data_path, phase, transform)

    def load_data(self) -> List[Dict]:
        """Load data from JSON files.
        
        Returns:
            List of data dictionaries
        """
        # Determine filename based on n_samples
        if self.phase == "train" and self.n_samples is not None:
            filename = f"train_data_perimage_subset_{self.n_samples}.json"
        else:
            filename = f"{self.phase}_data_perimage.json"
        
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise ValueError
            # Fallback to full dataset
            if self.phase == "train":
                filepath = os.path.join(self.data_path, "train_data_perimage.json")
                data = load_json(filepath)
                
                # Create subset if needed
                if self.n_samples is not None and len(data) > self.n_samples:
                    raise ValueError
                    random.shuffle(data)
                    data = data[:self.n_samples]
                
                return data
            else:
                filepath = os.path.join(self.data_path, f"{self.phase}_data_perimage.json")
        
        data = load_json(filepath)
        
        print(f"Loaded {len(data)} samples from {filepath}")
        return data

    def __getitem__(self, idx: int) -> Dict:
        """Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with sample data
        """
        sample = self.data[idx]
        
        img_path = sample["ImagePath"]
        findings = sample["Findings"].lower().strip()
        impression = sample.get("Impression", "").lower().strip()
        
        # Apply clustering transformations if needed
        # if self.phase == "train" and (self.use_random_from_cluster or self.use_expansion):
        #     findings = self._transform_findings(findings)

        if (self.use_random_from_cluster or self.use_expansion):
            findings = self._transform_findings(findings)
        
        # Prepare text
        if self.use_findings_only:
            full_text = "findings: " + findings
        else:
            full_text = "findings: " + findings + " impression: " + impression
        
        return {
            "img_path": img_path,
            "given_text": "findings:",
            "full_text": full_text,
            "findings": findings,
            "impression": impression,
        }

    def _transform_findings(self, findings: str) -> str:
        """Transform findings using clustering.
        
        Args:
            findings: Original findings text
            
        Returns:
            Transformed findings text
        """
        # Split into sentences
        sentences = split_sentences(findings)
        sentences = [preprocess_text(s) for s in sentences]
        
        # Random replacement from cluster

        if self.use_random_from_cluster:
            new_sentences = []
            for sentence in sentences:
                if sentence in self.thought_cluster:
                    cluster_id = self.thought_cluster[sentence]
                    replacement = random.choice(self.cluster_to_texts[cluster_id])
                    new_sentences.append(replacement)
                else:
                    new_sentences.append(sentence)
            sentences = new_sentences

        # Expansion
        if self.use_expansion:
            # Get cluster graph
            cluster_graph = []
            for sentence in sentences:
                if sentence in self.thought_cluster:
                    cluster_graph.append(self.thought_cluster[sentence])
                else:
                    cluster_graph.append(-1)
            
            cluster_graph = tuple(sorted([c for c in cluster_graph if c != -1]))
            
            # Get expansion 
            if cluster_graph in self.expansion_graphs:
                expanded_graph = random.choice(
                    self.expansion_graphs[cluster_graph]['expansions']
                )
                
                # Find added nodes
                added_nodes = set(expanded_graph) - set(cluster_graph)
                if added_nodes:
                    # Add sentences from added clusters
                    if not self.expansion_ablation:
                        added_sentences = [
                            random.choice(self.cluster_to_texts[node])
                            for node in added_nodes
                        ]
                    else:
                        added_sentences = [
                            random.choice(self.cluster_to_texts[random.choice(list(self.cluster_to_texts.keys()))])
                            for node in added_nodes
                        ]

                    sentences.extend(added_sentences)

            
        # Join sentences
        result = ". ".join(sentences)
        if not result.endswith("."):
            result += "."

        return result.strip()

    def _load_expansion_results(self, filepath: str) -> Dict:
        """Load expansion results from JSON.
        
        Args:
            filepath: Path to expansion file
            
        Returns:
            Dictionary mapping original graph to expansion info
        """
        with open(filepath, 'r') as f:
            json_list = json.load(f)
        
        result = {}
        for item in json_list:
            key = tuple(item['original_subgraph'])
            result[key] = {
                'count': item['count'],
                'expansions': [tuple(exp) for exp in item['expansions']]
            }
        
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        base_stats = super().get_statistics()
        
        # Count sentences
        n_sentences_list = []
        n_words_list = []
        
        for sample in self.data:
            findings = sample["Findings"]
            sentences = split_sentences(findings)
            n_sentences_list.append(len(sentences))
            n_words_list.append(len(findings.split()))
        
        base_stats.update({
            "use_findings_only": self.use_findings_only,
            "use_expansion": self.use_expansion,
            "use_random_from_cluster": self.use_random_from_cluster,
            "mean_sentences": float(np.mean(n_sentences_list)),
            "mean_words": float(np.mean(n_words_list)),
        })
        
        return base_stats


class RexGradDatasetFactory:
    """Factory for creating RexGrad datasets with different configurations."""

    @staticmethod
    def create(
        config,
        config_name: str,
        data_path: str,
        phase: str = "train",
        **kwargs
    ) -> RexGradDataset:
        """Create a dataset based on configuration name.
        
        Args:
            config_name: Configuration name
            data_path: Path to data directory
            phase: Dataset phase
            **kwargs: Additional arguments
            
        Returns:
            RexGradDataset instance
        """
        # Parse configuration name
        # Format: rexgrad[-only-findings][-{N}K][-cluster{K}-expansion]

        # use_findings_only = "only-findings" in config_name
        # use_expansion = "expansion" in config_name
        
        # # Extract n_samples
        # n_samples = None
        # if "10K" in config_name or "10k" in config_name.lower():
        #     n_samples = 10000
        # elif "50K" in config_name or "50k" in config_name.lower():
        #     n_samples = 50000
        # elif "100K" in config_name or "100k" in config_name.lower():
        #     n_samples = 100000
        
        # # Extract n_clusters
        # cluster_file = None
        # expansion_file = None
        
        # if "cluster" in config_name:
        #     import re
        #     match = re.search(r'cluster-?(\d+)', config_name.lower())
        #     if match:
        #         n_clusters = int(match.group(1))
                
        #         # Build file paths (update these based on your actual paths)
        #         if n_samples:
        #             cluster_file = f"{config.cluster_files_dir}/clusters_N{n_samples}_K{n_clusters}.json"
        #             expansion_file = f"{config.expansion_files_dir}/expansions_N{n_samples}_K{n_clusters}.json"
        
        # return RexGradDataset(
        #     data_path=data_path,
        #     phase=phase,
        #     n_samples=n_samples,
        #     use_findings_only=use_findings_only,
        #     use_expansion=use_expansion,
        #     cluster_file=cluster_file,
        #     expansion_file=expansion_file,
        #     **kwargs
        # )


        use_findings_only = "only-findings" in config_name
        use_expansion = "expansion" in config_name
        use_random_from_cluster = "random_from_cluster" in config_name
        expansion_ablation = "expansion_ablation" in config_name

        # Extract n_samples
        n_samples = None
        if "10K" in config_name or "10k" in config_name.lower():
            n_samples = 10000
        elif "50K" in config_name or "50k" in config_name.lower():
            n_samples = 50000
        elif "100K" in config_name or "100k" in config_name.lower():
            n_samples = 100000
        elif "300K" in config_name or "300k" in config_name.lower():
            n_samples = 300000

        # Extract clustering method
        cluster_method = None
        for method in ["kmeans", "hierarchical", "dbscan", "hdbscan"]:
            if f"cluster_method-{method}" in config_name.lower():
                cluster_method = method
                break

        # Extract n_clusters (only for kmeans and hierarchical)
        n_clusters = None
        cluster_file = None
        expansion_file = None

        if cluster_method:
            import re
            
            # For kmeans and hierarchical, extract n_clusters
            if cluster_method in ["kmeans", "hierarchical"]:
                match = re.search(r'ncluster-?(\d+)', config_name.lower())
                if match:
                    # n_clusters = int(match.group(1)) * 1000 
                    n_clusters = int(match.group(1)) 
                    #TODO fix this, this is a temporary solution 
                    if n_clusters < 500:
                        n_clusters *= 1000 
                    
                    if n_samples:
                        # Format: hierarchical_N100k_K1000_clusters.json
                        sample_str = f"{n_samples//1000}k" if n_samples >= 1000 else str(n_samples)
                        cluster_file = f"{config.cluster_files_dir}/{cluster_method}_N{sample_str}_K{n_clusters}_clusters.json"                      
                        if use_expansion:
                            expansion_file = f"{config.expansion_files_dir}/expansion_{cluster_method}_N{sample_str}_K{n_clusters}_expansion_results.json"
            
            # For dbscan and hdbscan, only use n_samples
            elif cluster_method in ["dbscan", "hdbscan"]:
                if n_samples:
                    # Format: dbscan_N100k_clusters.json (no K parameter)
                    sample_str = f"{n_samples//1000}k" if n_samples >= 1000 else str(n_samples)
                    cluster_file = f"{config.cluster_files_dir}/{cluster_method}_N{sample_str}_clusters.json"
                    
                    if use_expansion:
                        expansion_file = f"{config.expansion_files_dir}/expansion_{cluster_method}_N{sample_str}_expansion_results.json"

        return RexGradDataset(
            data_path=data_path,
            phase=phase,
            n_samples=n_samples,
            use_findings_only=use_findings_only,
            use_expansion=use_expansion,
            use_random_from_cluster=use_random_from_cluster,
            cluster_file=cluster_file,
            expansion_file=expansion_file,
            expansion_ablation=expansion_ablation,
            **kwargs
        )
