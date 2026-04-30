import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.features.sentiment import SentimentFeatures
from tqdm.auto import tqdm
from src.features.vectorizer import build_count_matrix


def main():
    # Load datasets
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")

    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    # Tokenization variants
    token_columns = [
        "tokens_cased",
        "tokens_lower",
        "tokens_letters",
        "tokens_filtered",
        "tokens_stemmed",
        "tokens_lemmatized"
    ]

    # Models to evaluate
    models = ["linear_svm", "logreg"]
    variants = ["CustomBow", "CustomTfidf"]

    # Z-score thresholds for vocabulary pruning
    z_scores = [0, 1, 2, 3, 4]

    # Total iterations for progress bar
    total_iters = len(token_columns) * len(z_scores) * len(models) * 2

    with tqdm(total=total_iters, desc="Feature Variants", ncols=100) as pbar:

        for token_col in token_columns:

            # Build base Count matrix once per tokenization
            X_train_base, X_val_base, _ = build_count_matrix(
                train_df[token_col],
                val_df[token_col]
            )

            # Fit sentiment feature statistics once
            sf = SentimentFeatures()
            sf.fit(X_train_base, y_train)

            # Iterate through Z-score pruning thresholds
            for z in z_scores:

                z_keep_indices = list(sf.filter_by_zscore(z))

                if not z_keep_indices:
                    print(f"\nWarning: Z-score {z} pruned all features for {token_col}. Skipping.")
                    pbar.update(len(models) * 2)
                    continue

                if len(z_keep_indices) < 2:
                    print(f"\nWarning: Z-score {z} left too few features for {token_col}. Skipping.")
                    pbar.update(len(models) * 2)
                    continue

                # Apply Z-score mask
                X_train_filtered = X_train_base[:, z_keep_indices]
                X_val_filtered = X_val_base[:, z_keep_indices]

                # Train models on filtered BoW
                for model_name in models:

                    model_save_name = f"{token_col}_{variants[0]}_Z{z}_{model_name}"

                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=model_save_name)
                    else:
                        clf = LogisticRegressionClassifier(name=model_save_name)

                    clf.train(X_train_filtered, y_train)
                    clf.evaluate(X_val_filtered, y_val, name="val/run_custom")

                    pbar.update(1)

                # TF-IDF transformation
                tfidf = TfidfTransformer()
                X_train_tfidf = tfidf.fit_transform(X_train_filtered)
                X_val_tfidf = tfidf.transform(X_val_filtered)

                # Train models on filtered TF-IDF
                for model_name in models:

                    model_save_name = f"{token_col}_{variants[1]}_Z{z}_{model_name}"

                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=model_save_name)
                    else:
                        clf = LogisticRegressionClassifier(name=model_save_name)

                    clf.train(X_train_tfidf, y_train)
                    clf.evaluate(X_val_tfidf, y_val, name="val/run_custom")

                    pbar.update(1)

                del X_train_filtered, X_val_filtered
                del X_train_tfidf, X_val_tfidf
                gc.collect()

            del X_train_base, X_val_base
            gc.collect()


if __name__ == "__main__":
    main()