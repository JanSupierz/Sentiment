from sklearn.feature_extraction.text import CountVectorizer


def space_tokenizer(text):
    """Split text exactly by spaces (bypass sklearn's regex)."""
    return text.split(' ')


def build_count_matrix(train_tokens, val_tokens, min_df=10, max_df=0.7, ngram_range=(1,3)):

    train_reviews = [" ".join(tokens) for tokens in train_tokens]
    val_reviews = None if val_tokens is None else [" ".join(tokens) for tokens in val_tokens]

    vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=space_tokenizer,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    X_train = vectorizer.fit_transform(train_reviews)
    X_val = None if val_tokens is None else vectorizer.transform(val_reviews)
    return X_train, X_val, vectorizer