import pandas as pd
import tensorflow_datasets as tfds
import re
import contractions
from sklearn.model_selection import train_test_split
import nltk

# Download sentence tokenizer if not already
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def tfds_to_df(ds):
    texts = []
    labels = []
    for text, label in ds:
        texts.append(text.numpy().decode('utf-8'))  # convert bytes to string
        labels.append(int(label.numpy()))           # convert tensor to int
    return pd.DataFrame({'review': texts, 'sentiment': labels})

def clean_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)           # Remove HTML tags
    text = contractions.fix(text)             # Expand contractions
    text = re.sub('[^a-zA-Z]', ' ', text)     # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def split_into_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    
    # Split into sentences
    raw_sentences = sent_tokenize(text)
    return [clean_text(s) for s in raw_sentences if clean_text(s)]

def split_into_words(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    
    # Split into words
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []

def load_reviews(unit_type: str, val_size=0.2, random_state=42):
    # Load IMDb dataset
    train_ds = tfds.load('imdb_reviews', split='train', as_supervised=True)
    test_ds = tfds.load('imdb_reviews', split='test', as_supervised=True)

    # Convert to DataFrame
    train_df = tfds_to_df(train_ds)
    test_df = tfds_to_df(test_ds)

    # Split reviews into sentences
    if(unit_type == "sentence"):
        train_df['units'] = train_df['review'].apply(split_into_sentences)
        test_df['units'] = test_df['review'].apply(split_into_sentences)
    elif(unit_type == "word"):
        train_df['units'] = train_df['review'].apply(split_into_words)
        test_df['units'] = test_df['review'].apply(split_into_words)
    else:
        raise ValueError("unit_type must be 'sentence' or 'word'")

    # Split train into train + validation
    test_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state, stratify=train_df['sentiment']
    )

    return train_df, test_df, val_df