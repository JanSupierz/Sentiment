import pandas as pd
import tensorflow_datasets as tfds
import re
import contractions
from sklearn.model_selection import train_test_split
import nltk
from IPython.display import display, Markdown

# Download sentence tokenizer if not already
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def tfds_to_df(ds):
    texts, labels = [], []
    for text, label in ds:
        texts.append(text.numpy().decode('utf-8'))
        labels.append(int(label.numpy()))
    return pd.DataFrame({'review': texts, 'sentiment': labels})

def clean_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = contractions.fix(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    raw_sentences = sent_tokenize(text)
    return [clean_text(s) for s in raw_sentences if clean_text(s)]

def split_into_words(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []

def load_reviews(unit_type: str, random_state=42, deduplicate=True):
    display(Markdown("## Loading IMDb Reviews Dataset"))

    # Load both splits
    train_ds = tfds.load('imdb_reviews', split='train', as_supervised=True)
    test_ds  = tfds.load('imdb_reviews', split='test', as_supervised=True)

    # Convert to DataFrames
    train_df = tfds_to_df(train_ds)
    test_df  = tfds_to_df(test_ds)

    # Concatenate into one DataFrame
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    display(Markdown(f"**Total reviews before deduplication:** {len(full_df)}"))

    if deduplicate:
        full_df = full_df.drop_duplicates(subset='review')
        display(Markdown(f"**Total reviews after deduplication:** {len(full_df)}"))

    # Split into train (60%) and test+val (40%)
    train_df, testval_df = train_test_split(
        full_df, test_size=0.4, random_state=random_state, stratify=full_df['sentiment']
    )

    # Split test+val into 50% validation, 50% test
    val_df, test_df = train_test_split(
        testval_df, test_size=0.5, random_state=random_state, stratify=testval_df['sentiment']
    )

    display(Markdown("**Train / Validation / Test sizes:**"))
    display(pd.DataFrame({
        'split': ['train', 'val', 'test'],
        'count': [len(train_df), len(val_df), len(test_df)]
    }))

    # Split reviews into units if requested
    if unit_type == "sentence":
        train_df['units'] = train_df['review'].apply(split_into_sentences)
        val_df['units'] = val_df['review'].apply(split_into_sentences)
        test_df['units'] = test_df['review'].apply(split_into_sentences)
    elif unit_type == "word":
        train_df['units'] = train_df['review'].apply(split_into_words)
        val_df['units'] = val_df['review'].apply(split_into_words)
        test_df['units'] = test_df['review'].apply(split_into_words)
    else:
        raise ValueError("unit_type must be 'sentence' or 'word'")

    # Show first 5 rows of each split
    display(Markdown("### Sample Reviews"))
    display(Markdown("**Train Set:**"))
    display(train_df.head())
    display(Markdown("**Test Set:**"))
    display(test_df.head())
    display(Markdown("**Validation Set:**"))
    display(val_df.head())

    return train_df, test_df, val_df
