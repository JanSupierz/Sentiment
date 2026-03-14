import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.features.sentiment import SentimentFeatures
from src.features.vectorizer import build_count_matrix
from tqdm.auto import tqdm

def run_ngram_experiment():
    print("Loading data...")
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    # Experiment parameters
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    models = ["linear_svm", "logreg"]
    total_iters = len(ngram_ranges) * len(models)

    with tqdm(total=total_iters, desc="Testing N-Gram Ranges", ncols=100) as pbar:
        for ngram in ngram_ranges:

            # Build Matrix
            X_train, X_val, _ = build_count_matrix(
                train_df['tokens_lower'], val_df['tokens_lower'],
                ngram_range=ngram, max_df=0.7
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

            # Train & Evaluate
            for model_key in models:
                model_name = f"ng{ngram[0]}-{ngram[1]}_{model_key}"

                if model_key == "linear_svm":
                    clf = LinearSVMClassifier(name=model_name)
                else:
                    clf = LogisticRegressionClassifier(name=model_name)

                clf.train(X_train, y_train)
                clf.evaluate(X_val, y_val, name="val/run_n_grams")
                pbar.update(1)

            # Cleanup memory
            del X_train, X_val
            gc.collect()


if __name__ == "__main__":
    run_ngram_experiment()