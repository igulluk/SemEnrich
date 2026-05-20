"""Script to run clustering on medical Impression."""

import argparse
import yaml
import json
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clustering import (
    KMeansClusterer,
    DBSCANClusterer,
    HierarchicalClusterer,
    HDBSCANClusterer,
    ClusterEvaluator,
)
from src.clustering.visualizer import ClusterVisualizer
from src.utils.io_utils import load_json, save_json, ensure_dir


def load_features_from_training_data(config):
    """Load features from training data NPZ files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (thought_dict, texts, vectors)
    """
    print("Loading training data...")
    
    data_path = config['data']['data_path']
    n_samples = config['data'].get('n_samples', None)
    feature_dir = config['data'].get('feature_dir', None)
    
    # Load metadata
    train_file = Path(data_path) / "train_data_perimage.json"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    all_data = load_json(train_file)
    
    # # Limit samples if specified
    # if n_samples and n_samples < len(all_data):
    #     import random
    #     random.seed(config['clustering']['random_seed'])
    #     all_data = random.sample(all_data, n_samples)
    #     subset_file = Path(data_path) / f"train_data_perimage_subset_{n_samples}.json"
    #     with open(subset_file, 'w') as file:
    #         json.dump(all_data, file)

    # Load features
    thought_dict = {}
    training_data = []
    
    for i, sample in enumerate(all_data):
        if i % 100 == 0:
            print(f"Loading sample {i+1}/{len(all_data)}...")
        
        img_path = sample['ImagePath']
        npz_path = img_path.replace("deid_png", "extracted_features").replace("png", "npz")
        try:
            data = np.load(npz_path)
            Impression = sample['Impression']
            
            # Split Impression into sentences
            if Impression.endswith("."):
                Impression_list = Impression.split(".")[:-1]
            else:
                Impression_list = Impression.split(".")
            
            Impression_list = [str(s).lower().strip() for s in Impression_list if str(s).strip()]
            embeddings = data['embeddings_impression']

            if len(Impression_list) != len(embeddings):
                continue

            removal_numerical_indices = set()
            numerical_set = set([str(x) for x in range(1,10)])
            for i in range(len(Impression_list)):
                if Impression_list[i] in numerical_set:
                    removal_numerical_indices.add(i)

            Impression_list = [item for i, item in enumerate(Impression_list) if i not in removal_numerical_indices]

            mask = np.ones(len(embeddings), dtype=bool)
            if removal_numerical_indices:
                mask[list(removal_numerical_indices)] = False
                embeddings = embeddings[mask]

            for text, embedding in zip(Impression_list, embeddings):
                thought_dict[text] = embedding
            
            training_data.append({
                'Impression': sample['Impression'],
                'embeddings_Impression': embeddings
            })
        
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            assert False 
            continue
        
        if n_samples and len(training_data) >= n_samples:
            break
    
    print(f"Loaded {len(thought_dict)} unique sentences from {len(training_data)} samples")
    
    # Convert to arrays
    texts = list(thought_dict.keys())
    vectors = np.array([thought_dict[t] for t in texts])
    
    return thought_dict, texts, vectors


def create_clusterer(config):
    """Create clusterer based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Clusterer instance
    """
    algorithm = config['clustering']['algorithm'].lower()
    random_seed = config['clustering']['random_seed']
    
    if algorithm == 'kmeans':
        return KMeansClusterer(
            n_clusters=config['clustering']['n_clusters'],
            max_iter=config['clustering'].get('max_iter', 300),
            n_init=config['clustering'].get('n_init', 10),
            random_seed=random_seed,
        )
    
    elif algorithm == 'dbscan':
        return DBSCANClusterer(
            eps=config['clustering']['eps'],
            min_samples=config['clustering']['min_samples'],
            metric=config['clustering'].get('metric', 'euclidean'),
            random_seed=random_seed,
        )
    
    elif algorithm == 'hierarchical':
        return HierarchicalClusterer(
            n_clusters=config['clustering']['n_clusters'],
            linkage=config['clustering'].get('linkage', 'ward'),
            metric=config['clustering'].get('metric', 'euclidean'),
            random_seed=random_seed,
        )
    
    elif algorithm == 'hdbscan':
        return HDBSCANClusterer(
            min_cluster_size=config['clustering'].get('min_cluster_size', 10),
            min_samples=config['clustering'].get('min_samples', None),
            metric=config['clustering'].get('metric', 'euclidean'),
            random_seed=random_seed,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main(config_path):
    """Main clustering workflow.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("CLUSTERING WORKFLOW")
    print("=" * 70)
    
    # Load data
    thought_dict, texts, vectors = load_features_from_training_data(config)
    
    # Create clusterer
    print("\nInitializing clusterer...")
    clusterer = create_clusterer(config)
    
    # Fit clustering
    print("\nFitting clustering algorithm...")
    clusterer.fit(vectors)
    labels = clusterer.labels_
    
    print(f"Found {clusterer.n_clusters_} clusters")
    
    # Get mappings
    print("\nCreating cluster mappings...")
    text_to_cluster = clusterer.get_cluster_mapping(texts, vectors)
    cluster_to_texts = clusterer.get_cluster_to_texts(texts, vectors)
    
    # Evaluate
    print("\nEvaluating clustering...")
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate(vectors, labels, config['evaluation']['metrics'])
    evaluator.print_report(vectors, labels)
    
    # Save outputs
    output_dir = ensure_dir(config['output']['output_dir'])
    prefix = config['output']['save_prefix']
    
    if config['output'].get('save_cluster_mapping', True):
        output_file = output_dir / f"{prefix}_clusters.json"
        save_json(text_to_cluster, output_file)
    
    if config['output'].get('save_cluster_to_texts', True):
        output_file = output_dir / f"{prefix}_cluster_to_texts.json"
        save_json(cluster_to_texts, output_file)
    
    if config['output'].get('save_metrics', True):
        output_file = output_dir / f"{prefix}_metrics.json"
        save_json(metrics, output_file)
        
        # Save cluster statistics
        stats = clusterer.get_cluster_statistics()
        output_file = output_dir / f"{prefix}_statistics.json"
        save_json(stats, output_file)
    
    # Visualizations
    if config['evaluation'].get('visualization', False):
        print("\nGenerating visualizations...")
        visualizer = ClusterVisualizer()
        
        for method in config['evaluation'].get('visualization_methods', ['tsne']):
            visualizer.plot_2d_projection(
                vectors, labels, method=method,
                save_path=str(output_dir / f"{prefix}_2d_{method}.png")
            )
        
        visualizer.plot_cluster_sizes(
            labels,
            save_path=str(output_dir / f"{prefix}_sizes.png")
        )
        
        visualizer.plot_silhouette(
            vectors, labels,
            save_path=str(output_dir / f"{prefix}_silhouette.png")
        )
    
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering on medical Impression")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    main(args.config)

