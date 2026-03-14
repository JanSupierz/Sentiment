#!/usr/bin/env python3
import argparse
import yaml
import itertools
from sklearn.feature_extraction.text import TfidfTransformer
import dask
from dask.distributed import Client, LocalCluster, as_completed
from tqdm import tqdm

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.utils.paths import PROJECT_ROOT
from src.features import builder


def run_grid_iter(token_col, n_concepts, sentiment_weight, model_name, static_cfg):
    run_id = f"{model_name}_k{n_concepts}_w{int(sentiment_weight)}"

    try:
        X_train, y_train = builder.load_representation(
            token_col, n_concepts, sentiment_weight, 2, 'train'
        )
        X_val, y_val = builder.load_representation(
            token_col, n_concepts, sentiment_weight, 2, 'val'
        )

        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)

        if model_name == 'linear_svm':
            model = LinearSVMClassifier(name=run_id)
        else:
            model = LogisticRegressionClassifier(name=run_id)

        model.train(X_train_tfidf, y_train)
        model.evaluate(X_val_tfidf, y_val, name="val/grid_search")

        return f"Done: {run_id}"
    except Exception as e:
        return f"Failed {run_id}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--token_col", default="tokens_lower", help="Token column to use")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        static_cfg = yaml.safe_load(f)
    grid = static_cfg['grid_search']

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    token_col = args.token_col

    delayed_tasks = []
    for nc, w, m in itertools.product(
        grid['n_concepts'], grid['sentiment_weight'], grid['models']
    ):
        if nc == 0 and w > 0:
            continue
        delayed_tasks.append(dask.delayed(run_grid_iter)(
            token_col, nc, w, m, static_cfg
        ))

    if delayed_tasks:
        print(f"Starting Grid Search with {len(delayed_tasks)} iterations...")
        futures = client.compute(delayed_tasks)

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search Progress"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"Failed: {e}")

        failed = [r for r in results if r.startswith("Failed")]
        print(f"\nCompleted: {len(results) - len(failed)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for f in failed[:5]: print(f"  {f}")

    client.close()


if __name__ == "__main__":
    main()