#!/usr/bin/env python3
import argparse
import yaml
import itertools
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfTransformer
import dask
from dask.distributed import Client, LocalCluster, as_completed
from tqdm import tqdm

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.features.concept_remap import remap_sparse_matrix
from src.utils.paths import PROJECT_ROOT, DATA_DIR
from src.features.builder import load_representation

def run_grid_iter(static_cfg, nr, nc, w, z, m):
    # Construct a unique, descriptive name
    run_id = f"{m}_ng{nr[0]}-{nr[1]}_k{nc}_w{int(w)}_z{z}"
    
    try:
        X_train, y_train = load_representation(static_cfg, nr, nc, w, z, 'train')
        X_val, y_val = load_representation(static_cfg, nr, nc, w, z, 'val')
        
        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)
        
        if m == 'linear_svm':
            model = LinearSVMClassifier(C=static_cfg['grid_search']['C'][0], name=run_id)
        else:
            model = LogisticRegressionClassifier(C=static_cfg['grid_search']['C'][0], name=run_id)
            
        model.train(X_train_tfidf, y_train)
        model.evaluate(X_val_tfidf, y_val, name="val")
        
        return f"Done: {run_id}"
    except Exception as e:
        return f"Failed {run_id}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        static_cfg = yaml.safe_load(f)
    grid = static_cfg['grid_search']

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    delayed_tasks = []
    for nr, nc, z, w, m in itertools.product(
        grid['ngram_range'], grid['n_concepts'], grid['z_threshold'], 
        grid['sentiment_weight'], grid['models']
    ):
        if nc == 0 and w > 0: continue # Deduplicate
        delayed_tasks.append(dask.delayed(run_grid_iter)(static_cfg, nr, nc, w, z, m))
    
    if delayed_tasks:
        print(f"🚀 Starting Grid Search with {len(delayed_tasks)} iterations...")
        futures = client.compute(delayed_tasks)
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search Progress"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"Failed: {e}")

        failed = [r for r in results if r.startswith("Failed")]
        print(f"\n✅ Completed: {len(results) - len(failed)}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
            for f in failed[:5]: print(f"  {f}")

    client.close()

if __name__ == "__main__":
    main()