import pandas as pd
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from tqdm.auto import tqdm


def main():
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    models = ["linear_svm", "logreg"]
    text_columns = ["text_clean", "text_expanded"]
    casing_options = [True, False]

    total_iters = len(text_columns) * len(casing_options) * 2 * len(models)

    with tqdm(total=total_iters, desc="Training & evaluating", ncols=100) as pbar:

        for text_col in text_columns:
            for is_lower in casing_options:
                # Create a label for saving the models/metrics
                case_label = "lower" if is_lower else "cased"

                vectorizer = CountVectorizer(
                    ngram_range=(1, 3),
                    min_df=10,
                    max_df=0.7,
                    lowercase=is_lower
                )

                # Extract features for the current text column and casing
                X_train = vectorizer.fit_transform(train_df[text_col])
                X_val = vectorizer.transform(val_df[text_col])

                # --- BoW Phase ---
                for model_name in models:
                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=f"{text_col}_{case_label}_BoW_{model_name}")
                    else:
                        clf = LogisticRegressionClassifier(name=f"{text_col}_{case_label}_BoW_{model_name}")

                    clf.train(X_train, y_train)
                    clf.evaluate(X_val, y_val, name="val/run_basic")
                    pbar.update(1)

                # --- TF-IDF Phase ---
                tfidf = TfidfTransformer()
                X_train = tfidf.fit_transform(X_train)
                X_val = tfidf.transform(X_val)
                gc.collect()

                for model_name in models:
                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=f"{text_col}_{case_label}_TF-IDF_{model_name}")
                    else:
                        clf = LogisticRegressionClassifier(name=f"{text_col}_{case_label}_TF-IDF_{model_name}")

                    clf.train(X_train, y_train)
                    clf.evaluate(X_val, y_val, name="val/run_basic")
                    pbar.update(1)

                del X_train, X_val
                gc.collect()


if __name__ == "__main__":
    main()