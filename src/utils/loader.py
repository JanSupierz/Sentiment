import os
import warnings
import logging
import json
import re
import unicodedata
import contractions
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import argparse
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.object.*")
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.utils.visualizer import ModelVisualizer
from src.utils.paths import DATA_DIR

class DataLoader:
    TRANSLATE_TABLE = str.maketrans({
        "`": "'", "´": "'", "’": "'", "‘": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "-", "−": "-",
        "\x96": "-", "\x97": "-",
        "…": "..."
    })

    @classmethod
    def process_single_item(cls, item: Tuple[bytes, int]) -> Dict:
        """Clean and tokenise a single review."""
        text_raw, label = item
        text_raw = text_raw.decode("utf-8") if isinstance(text_raw, bytes) else text_raw

        text = unicodedata.normalize('NFKC', text_raw)
        text = text.translate(cls.TRANSLATE_TABLE)
        text = re.sub(r'<[^>]+>', ' ', text)
        clean_bert = re.sub(r'\s+', ' ', text).strip()
        clean_bow = contractions.fix(clean_bert.lower())

        return {
            "review": text_raw,
            "clean_review": clean_bert,
            "clean_bow": clean_bow,
            "sentiment": int(label)
        }

    @staticmethod
    def get_ngrams(text: str, ngram_range: Tuple[int, int] = (1, 3)) -> List[str]:
        """Extract n‑grams (word‑level) from pre‑tokenised BoW text."""
        tokens = re.findall(r'\w+|[^\w\s]', text)
        min_n, max_n = ngram_range
        units = []
        for n in range(min_n, max_n + 1):
            if n == 1:
                units.extend(tokens)
            else:
                grams = zip(*[tokens[i:] for i in range(n)])
                units.extend([" ".join(g) for g in grams])
        return units

    @classmethod
    def load_imdb(cls, n_jobs: int = 4, train_size: float = 0.6, test_size: float = 0.8) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load IMDb reviews, clean, deduplicate, split.
        Returns (train, val, test) as lists of dicts.
        """
        save_dir = str(DATA_DIR / "IMDb")

        print("Loading IMDb reviews...")
        ds = tfds.load('imdb_reviews', split='train+test', as_supervised=True)
        raw_data = list(tfds.as_numpy(ds))

        seen = set()
        unique_raw = []
        for t, l in raw_data:
            if t not in seen:
                seen.add(t)
                unique_raw.append((t, l))

        print(f"Found {len(raw_data)-len(unique_raw)} duplicates.")
        print(f"Cleaning {len(unique_raw)} reviews...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            processed = list(tqdm(executor.map(cls.process_single_item, unique_raw), total=len(unique_raw)))

        labels = [item['sentiment'] for item in processed]
        train, temp_data, _, temp_labels = train_test_split(
            processed, labels, test_size=1 - train_size, random_state=42, stratify=labels
        )
        val, test = train_test_split(
            temp_data, test_size=test_size, random_state=42, stratify=temp_labels
        )

        print(f"Loaded dataset with {len(train)} train, {len(val)} val and {len(test)} test samples.")

        os.makedirs(save_dir, exist_ok=True)
        for name, dataset in zip(["train", "val", "test"], [train, val, test]):
            path = os.path.join(save_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Loading done. JSON files saved to {save_dir}")

        return train, val, test

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    train, val, test = DataLoader.load_imdb(
        train_size=cfg['data']['train_size'],
        test_size=cfg['data']['test_size']
    )

    def to_df(data):
        df = pd.DataFrame(data)
        if 'review' in df.columns:
            df.drop(columns=['review'], inplace=True)
        return df

    train_df = to_df(train)
    val_df   = to_df(val)
    test_df  = to_df(test)

    # Save as Parquet
    out_dir = DATA_DIR / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_dir / "train.parquet")
    val_df.to_parquet(out_dir / "val.parquet")
    test_df.to_parquet(out_dir / "test.parquet")

    print(f"Preprocessing done. Parquet files saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)