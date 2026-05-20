import numpy as np 
import json 
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
from PIL import Image 
import torch 
import os 
import random 
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
from transformers import ViTModel
import time
import sys 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Get current time as seed (in milliseconds for more variation)
seed = int(time.time() * 1000000) % (2**32)  # Keep it within 32-bit range
seed = sum([int(x) for x in str(seed)])
print(f'SEED : {seed}')

# Set all random seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU


def process_sample(sample, sentence_model_name, dino_model_name):
    """
    Process a single sample to extract and save features.
    Each process loads its own model instances to avoid sharing issues.
    """
    # Load models in each worker process
    model = SentenceTransformer(sentence_model_name)
    # dino_model = AutoModel.from_pretrained(dino_model_name)

    # Load without the pooler layer
    dino_model = ViTModel.from_pretrained(
        dino_model_name,
        add_pooling_layer=False  # This prevents the warning
    )

    dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
    dino_model.eval()

    npz_file_path = sample['ImagePath'].replace("deid_png", "extracted_features").replace("png", "npz")

    # Check if file already exists (optional: skip if already processed)
    if os.path.exists(npz_file_path):
        # print(f"Skipped (already exists): {npz_file_path}")
        return f"Skipped (already exists): {npz_file_path}"    
    print(f"Starting Processing: {npz_file_path}")

    try:
        # Process Findings
        sentences_findings = [x if x.endswith(".") else x + "." for x in sample['Findings'].split(". ")]
        embeddings_findings = model.encode(sentences_findings)
        embeddings_findings = np.array(embeddings_findings)
        norms = np.linalg.norm(embeddings_findings, axis=1, keepdims=True)
        embeddings_findings = embeddings_findings / norms
        
        # Process Impression
        sentences_impression = [x if x.endswith(".") else x + "." for x in sample['Impression'].split(". ")]
        embeddings_impression = model.encode(sentences_impression)
        embeddings_impression = np.array(embeddings_impression)
        norms = np.linalg.norm(embeddings_impression, axis=1, keepdims=True)
        embeddings_impression = embeddings_impression / norms
        
        # Process Image
        image = Image.open(sample['ImagePath']).convert("RGB")        
        inputs = dino_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = dino_model(**inputs)
        
        # Get the [CLS] token as the image feature
        image_features = outputs.last_hidden_state[:, 0, :]
        norm = np.linalg.norm(image_features.cpu().numpy())
        image_features = image_features.cpu().numpy() / norm
        
        # Save features
        os.makedirs(os.path.dirname(npz_file_path), exist_ok=True)
        np.savez_compressed(
            npz_file_path, 
            embeddings_findings=embeddings_findings.astype(np.float32),
            embeddings_impression=embeddings_impression.astype(np.float32),
            image_features=image_features.astype(np.float32),
        )
        print(f"Processed: {npz_file_path}")
        return f"Processed: {npz_file_path}"
    
    except Exception as e:
        # print(f"Error processing {sample['ImagePath']}: {str(e)}")
        return f"Error processing {sample['ImagePath']}: {str(e)}"


def main():
    # Configuration
    sentence_model_name = "all-MiniLM-L6-v2"
    dino_model_name = "facebook/dino-vits16"
    # num_workers = cpu_count() - 1  # Leave one CPU free
    num_workers = 2
    # Alternatively, set a specific number of workers
    # num_workers = 8
    
    print(f"Using {num_workers} worker processes")
    
    DATA_PATH = "./data"
    OUTPUT_PATH = "./extracted_features"
    # Load data
    print("Loading data...")
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    
        
    from pathlib import Path
    directory = Path(OUTPUT_PATH)
    already_extracted_files = [str(f) for f in directory.rglob('*') if f.is_file()]
    already_extracted_files = set(already_extracted_files)
    data = [x for x in data if x['ImagePath'].replace("deid_png", "extracted_features").replace("png", "npz") not in already_extracted_files]
    print(f'len of subset of the data : {len(data)}')
    random.shuffle(data) 
    print(f"Total samples to process: {len(data)}")
    
    # Create partial function with fixed model names
    process_func = partial(process_sample, 
                          sentence_model_name=sentence_model_name,
                          dino_model_name=dino_model_name)
    
    # Process in parallel with progress bar
    print("Processing samples in parallel...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, data),
            total=len(data),
            desc="Processing samples"
        ))
    
    # Print summary
    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)
    
    # Count results
    processed = sum(1 for r in results if r.startswith("Processed:"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    errors = sum(1 for r in results if r.startswith("Error"))
    
    print(f"Successfully processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    
    # Print errors if any
    if errors > 0:
        print("\nErrors encountered:")
        for r in results:
            if r.startswith("Error"):
                print(f"  - {r}")


if __name__ == "__main__":
    main()

