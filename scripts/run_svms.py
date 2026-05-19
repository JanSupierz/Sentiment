import pandas as pd
import gc
import time
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, RbfSVMClassifier
from src.features.sentiment import SentimentFeatures
from src.features.vectorizer import build_count_matrix
from tqdm.auto import tqdm


def run_svm_experiment():
    print("Loading data...")
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    # Experiment parameters
    models = ["linear_svm", "rbf_svm"]
    total_iters = len(models)

    with tqdm(total=total_iters, desc="Testing svm kernels", ncols=100) as pbar:
        # Build Matrix
        X_train, X_val, _ = build_count_matrix(
            train_df['tokens_lower'], val_df['tokens_lower'],
            ngram_range=(1,3)
        )

        # Apply Z-Score Pruning
        sf = SentimentFeatures()
        sf.fit(X_train, y_train.tolist())
        z_keep_indices = list(sf.filter_by_zscore(2.0))

        X_train = X_train[:, z_keep_indices]
        X_val = X_val[:, z_keep_indices]

        # TF-IDF Transformation
        tfidf = TfidfTransformer()
        X_train = tfidf.fit_transform(X_train)
        X_val = tfidf.transform(X_val)

        max_iter = 1000
        
        # Train & Evaluate
        for model_key in models:
            model_name = f"ng1-3_{model_key}_{max_iter}"

            if model_key == "linear_svm":
                clf = LinearSVMClassifier(name=model_name)
            else:
                clf = RbfSVMClassifier(name=model_name, max_iter=max_iter)

            start_time = time.perf_counter()
            clf.train(X_train, y_train)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            tqdm.write(f"---> Training {model_key} took: {elapsed_time:.2f} seconds")

            clf.evaluate(X_val, y_val, name="val/run_svms")
            pbar.update(1)

        # Cleanup memory
        del X_train, X_val
        gc.collect()


if __name__ == "__main__":
    run_svm_experiment()