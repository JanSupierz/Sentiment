import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any

import contractions
import nltk
import pandas as pd
import spacy
import tensorflow_datasets as tfds
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def get_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    """Generate n-grams from a list of tokens."""
    min_n, max_n = ngram_range
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


TRANSLATE_TABLE = str.maketrans({
    "`": "'", "´": "'", "’": "'", "‘": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "\x96": "-", "\x97": "-",
    "…": "..."
})


def normalize_text(text: str) -> str:
    """Unify punctuation, normalise Unicode, strip HTML and extra spaces."""
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(TRANSLATE_TABLE)
    text = re.sub(r'<[^>]+>', ' ', text)         # remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_letters(tokens: List[str]) -> List[str]:
    """Remove all non letter characters from tokens, discard if empty."""
    cleaned = []
    for t in tokens:
        token_clean = re.sub(r'[^a-zA-Z]', '', t)
        if token_clean:
            cleaned.append(token_clean)
    return cleaned


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply Porter stemming to each token."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Apply WordNet lemmatization to each token."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def process_one(item: Tuple[bytes, int]) -> Dict[str, Any]:
    """Clean a single review and produce all text representations."""
    text_raw, label = item
    if isinstance(text_raw, bytes):
        text_raw = text_raw.decode('utf-8')

    text_clean = normalize_text(text_raw)
    text_expanded = contractions.fix(text_clean)

    tokens_cased = word_tokenize(text_expanded)
    tokens_lower = [w.lower() for w in tokens_cased]

    doc = nlp(text_expanded)
    tokens_filtered = [
        token.text.lower() for token in doc
        if (token.pos_ != 'AUX' or token.tag_ == 'MD')
        and token.pos_ != 'DET'
        and token.tag_ != 'POS'
    ]

    return {
        'sentiment': int(label),
        'text_raw': text_raw,
        'text_clean': text_clean,
        'text_expanded': text_expanded,
        'tokens_cased': tokens_cased,
        'tokens_lower': tokens_lower,
        'tokens_letters': clean_letters(tokens_lower),
        'tokens_stemmed': stem_tokens(tokens_lower),
        'tokens_lemmatized': lemmatize_tokens(tokens_lower),
        'tokens_filtered': tokens_filtered
    }


def load_and_process(n_jobs: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    print("Loading IMDb reviews...")
    ds = tfds.load('imdb_reviews', split='train+test', as_supervised=True)
    raw_data = list(tfds.as_numpy(ds))

    # Deduplicate based on raw review text
    seen = set()
    unique_raw = []
    for t, l in raw_data:
        if t not in seen:
            seen.add(t)
            unique_raw.append((t, l))
    print(f"Removed {len(raw_data)-len(unique_raw)} duplicates.")

    print(f"Processing {len(unique_raw)} reviews...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        processed = []
        for result in tqdm(executor.map(process_one, unique_raw), total=len(unique_raw)):
            processed.append(result)

    labels = [item['sentiment'] for item in processed]

    train, temp, _, temp_labels = train_test_split(
        processed, labels, test_size=0.5,
        random_state=42, stratify=labels
    )

    val, test = train_test_split(
        temp, test_size=0.3,
        random_state=42, stratify=temp_labels
    )

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def save_as_parquet(train, val, test, out_dir):
    """Convert lists to DataFrames and save as Parquet."""
    os.makedirs(out_dir, exist_ok=True)

    for name, data in zip(['train', 'val', 'test'], [train, val, test]):
        df = pd.DataFrame(data)

        df = df[['sentiment', 'text_raw', 'text_clean', 'text_expanded',
                 'tokens_cased', "tokens_lower", 'tokens_filtered',
                 'tokens_letters', 'tokens_stemmed', 'tokens_lemmatized']]

        out_path = os.path.join(out_dir, f'{name}.parquet')
        df.to_parquet(out_path, index=False)
        print(f"Saved {name} set to {out_path}")


if __name__ == '__main__':
    save_as_parquet(*load_and_process(n_jobs=4), out_dir='data/preprocessed')