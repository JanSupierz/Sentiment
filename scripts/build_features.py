#!/usr/bin/env python3
import argparse
import yaml
from tqdm import tqdm
from src.utils.paths import PROJECT_ROOT
from src.features import builder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    grid = cfg['grid_search']

    print("--- PHASE 1: Building Base Vocabulary & Embeddings ---")
    unique_ngrams = grid['ngram_range']
    
    for nr in tqdm(unique_ngrams, desc="Base Setup", dynamic_ncols=True):
        builder.build_ngram_index(cfg, nr, splits=['train', 'val'])
        builder.compute_and_cache_embeddings(cfg, nr)

    print("\n--- PHASE 2: Extracting Concepts & Pre-building Matrices ---")
    tasks = []
    for nr in grid['ngram_range']:
        for nc in grid['n_concepts']:
            weights = [0.0] if nc == 0 else grid['sentiment_weight']
            for w in weights:
                # Deduplicate baseline tasks
                if nc == 0 and w > 0:
                    continue
                tasks.append((tuple(nr), nc, w))

    # Remove any stray duplicates
    tasks = list(dict.fromkeys(tasks))

    with tqdm(total=len(tasks), desc="Processing Configurations", dynamic_ncols=True) as pbar:
        for nr, nc, w in tasks:
            pbar.set_description(f"ng{nr[0]}-{nr[1]} | k{nc} | w{int(w)}")
            
            if nc > 0:
                # 1. Cluster embeddings into concepts
                builder.run_extraction_logic(nr, nc, w)
                
                # 2. Pre-build concept matrices (resolves unseen validation words now)
                builder.build_concept_matrices(cfg, nr, nc, w, splits=['train', 'val'])
            
            # 3. Compute log-odds stats (relies on the train matrix)
            builder.run_stats_logic(cfg, nr, nc, w)
            
            pbar.update(1)

    print("\nFeature Factory Complete! Matrices cached. Ready for parallel grid search.")

if __name__ == "__main__":
    main()