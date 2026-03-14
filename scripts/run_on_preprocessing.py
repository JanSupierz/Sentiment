import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from tqdm.auto import tqdm
from src.features.vectorizer import build_count_matrix


def main():
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    token_columns = [
        'tokens_cased', 'tokens_lower', 'tokens_letters', 'tokens_filtered',
        'tokens_stemmed', 'tokens_lemmatized'
    ]
    models = ["linear_svm", "logreg"]

    # 2 represents BoW and TF-IDF phases
    total_iters = len(token_columns) * 2 * len(models)

    with tqdm(total=total_iters, desc="Training & evaluating", ncols=100) as pbar:
        for token_col in token_columns:

            X_train, X_val, _ = build_count_matrix(train_df[token_col], val_df[token_col])

            for model_name in models:
                if model_name == "linear_svm":
                    clf = LinearSVMClassifier(name=f"{token_col}_BoW_{model_name}")
                else:
                    clf = LogisticRegressionClassifier(name=f"{token_col}_BoW_{model_name}")

                clf.train(X_train, y_train)
                clf.evaluate(X_val, y_val, name="val/run_on_preprocessing")
                pbar.update(1)

            tfidf = TfidfTransformer()
            X_train = tfidf.fit_transform(X_train)
            X_val = tfidf.transform(X_val)
            gc.collect()

            for model_name in models:
                if model_name == "linear_svm":
                    clf = LinearSVMClassifier(name=f"{token_col}_TF-IDF_{model_name}")
                else:
                    clf = LogisticRegressionClassifier(name=f"{token_col}_TF-IDF_{model_name}")

                clf.train(X_train, y_train)
                clf.evaluate(X_val, y_val, name="val/run_on_preprocessing")
                pbar.update(1)

            del X_train, X_val
            gc.collect()


if __name__ == "__main__":
    main()