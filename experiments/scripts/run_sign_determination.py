"""Script to determine cluster signs (positive/negative)."""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clustering.sign_determiner import ClusterSignDeterminer
from src.utils.io_utils import load_json, save_json, ensure_dir


def main(config_path):
    """Main sign determination workflow.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("CLUSTER SIGN DETERMINATION")
    print("=" * 70)
    
    # Load cluster data
    print("\nLoading cluster data...")
    cluster_to_texts = load_json(config['clustering']['cluster_to_texts_file'])
    
    # Convert keys to int
    cluster_to_texts = {int(k): v for k, v in cluster_to_texts.items()}
    
    # Determine signs
    method = config['sign_determination']['method']
    
    if method == 'llm':
        provider = config['sign_determination']['llm']['provider']
        print(f"\nUsing LLM-based sign determination with {provider}...")
        
        api_key = config['sign_determination']['llm'].get('api_key')
        if api_key and api_key.startswith('${'):
            # Get from environment variable
            import os
            env_var = api_key.strip('${}')
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not set. "
                               f"Please set it: export {env_var}='your-api-key'")
        
        determiner = ClusterSignDeterminer(
            api_key=api_key,
            provider=provider,
            model=config['sign_determination']['llm']['model'],
            temperature=config['sign_determination']['llm'].get('temperature', 0.9),
        )
        
        cluster_signs = determiner.determine_signs(cluster_to_texts)
    
    elif method == 'rule_based':
        print("\nUsing rule-based sign determination...")
        print("(Faster but less accurate than LLM-based approach)\n")
        
        determiner = ClusterSignDeterminer()
        cluster_signs = determiner.determine_signs_rule_based(cluster_to_texts)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'llm' or 'rule_based'")
    
    # Save results
    output_dir = ensure_dir(config['output']['output_dir'])
    prefix = config['output']['save_prefix']
    
    if config['output'].get('save_signs', True):
        output_file = output_dir / f"{prefix}_signs.json"
        save_json(cluster_signs, output_file)
    
    # Save examples
    if config['output'].get('save_examples', True):
        n_examples = config['output'].get('n_examples', 10)
        
        # Get examples for positive and negative clusters
        positive_examples = determiner.get_cluster_examples(
            cluster_to_texts, cluster_signs,
            n_examples=n_examples, sign_filter='positive'
        )
        
        negative_examples = determiner.get_cluster_examples(
            cluster_to_texts, cluster_signs,
            n_examples=n_examples, sign_filter='negative'
        )
        
        output_file = output_dir / f"{prefix}_positive_examples.json"
        save_json(positive_examples, output_file)
        
        output_file = output_dir / f"{prefix}_negative_examples.json"
        save_json(negative_examples, output_file)
        
        # Print some examples
        print("\n" + "=" * 70)
        print("EXAMPLE POSITIVE CLUSTERS")
        print("=" * 70)
        
        for cluster_id in list(positive_examples.keys())[:3]:
            print(f"\nCluster {cluster_id}:")
            for text in positive_examples[cluster_id][:3]:
                print(f"  - {text}")
        
        print("\n" + "=" * 70)
        print("EXAMPLE NEGATIVE CLUSTERS")
        print("=" * 70)
        
        for cluster_id in list(negative_examples.keys())[:3]:
            print(f"\nCluster {cluster_id}:")
            for text in negative_examples[cluster_id][:3]:
                print(f"  - {text}")
    
    print("\n" + "=" * 70)
    print("SIGN DETERMINATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine cluster signs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    main(args.config)

