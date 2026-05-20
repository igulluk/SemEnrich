"""Script to generate findings expansions."""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clustering.expansion import FindingsExpander
from src.utils.io_utils import load_json, save_json, ensure_dir
import numpy as np


def save_expansion_results_to_json(expansion_results, filename):
    """Save expansion results to JSON format.
    
    Args:
        expansion_results: Dictionary of expansion results
        filename: Output filename
    """
    json_list = []
    for original_graph, result in expansion_results.items():
        json_list.append({
            'original_subgraph': list([int(k) for k in original_graph]),
            'count': str(result['count']),
            'expansions': [tuple([int(k) for k in exp]) for exp in result['expansions']]
        })
    save_json(json_list, filename)


def main(config_path):
    """Main expansion generation workflow.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("FINDINGS EXPANSION GENERATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    
    data_path = config['data']['data_path']
    n_samples = config['data'].get('n_samples', None)
    
    # Load training data
    train_file = Path(data_path) / "train_data_perimage.json"
    if n_samples:
        subset_file = Path(data_path) / f"train_data_perimage_subset_{n_samples}.json"
        if subset_file.exists():
            train_file = subset_file
    
    training_data = load_json(train_file)
    
    if n_samples and len(training_data) > n_samples:
        assert False 
        import random
        random.seed(42)
        training_data = random.sample(training_data, n_samples)
    
    print(f"Loaded {len(training_data)} training samples")
    
    # Load cluster data
    thought_cluster = load_json(config['clustering']['cluster_file'])
    cluster_signs = load_json(config['clustering']['cluster_signs_file'])
    
    # Convert signs keys to int
    cluster_signs = {int(k): v for k, v in cluster_signs.items()}
    
    print(f"Loaded {len(thought_cluster)} text-to-cluster mappings")
    print(f"Loaded {len(cluster_signs)} cluster signs")
    
    # Create expander
    expander = FindingsExpander(
        adjacency_threshold_norm=config['expansion']['adjacency_threshold_norm'],
        adjacency_threshold_count=config['expansion']['adjacency_threshold_count'],
        only_positive_expansion=config['expansion']['only_positive_expansion'],
    )
    
    # Generate expansions
    print("\nGenerating expansions...")
    expansion_results = expander.generate_expansions(
        training_data,
        thought_cluster,
        cluster_signs
    )
    
    # Get statistics
    stats = expander.get_expansion_statistics(expansion_results)
    
    print("\n" + "=" * 70)
    print("EXPANSION STATISTICS")
    print("=" * 70)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save outputs
    output_dir = ensure_dir(config['output']['output_dir'])
    prefix = config['output']['save_prefix']
    
    if config['output'].get('save_expansion_results', True):
        output_file = output_dir / f"{prefix}_expansion_results.json"
        save_expansion_results_to_json(expansion_results, output_file)
    
    if config['output'].get('save_adjacency_matrix', True):
        output_file = output_dir / f"{prefix}_adjacency.npz"
        np.savez(
            output_file,
            adjacency=expander.adjacency_matrix,
            is_addable=expander.is_addable
        )
        print(f"Saved adjacency matrices to {output_file}")
    
    if config['output'].get('save_statistics', True):
        output_file = output_dir / f"{prefix}_statistics.json"
        save_json(stats, output_file)
    
    print("\n" + "=" * 70)
    print("EXPANSION GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate findings expansions")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    main(args.config)

