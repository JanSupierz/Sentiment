import os
import json
import re
import unicodedata
import contractions
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from src.utils.visualizer import ModelVisualizer

class DataLoader:
    TRANSLATE_TABLE = str.maketrans({
        "`": "'", "´": "'", "’": "'", "‘": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "-", "−": "-",
        "\x96": "-", "\x97": "-",
        "…": "..."
    })

    @classmethod
    def process_single_item(cls, item):
        text_raw, label = item
        text_raw = text_raw.decode("utf-8") if isinstance(text_raw, bytes) else text_raw
        
        # 1. Clean for BERT and BoW
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
    def get_ngrams(text, ngram_range=(1, 3)):
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
    def load_imdb(cls, n_jobs=4, train_size=0.6, test_size=0.8, save_dir="data/IMDb"):
        print("Loading IMDb reviews...")
        ds = tfds.load('imdb_reviews', split='train+test', as_supervised=True)
        raw_data = list(tfds.as_numpy(ds))
        
        # Remove duplicates
        seen = set()
        unique_raw = [(t, l) for t, l in raw_data if not (t in seen or seen.add(t))]
        
        print(f"Cleaning {len(unique_raw)} reviews...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            processed = list(tqdm(executor.map(cls.process_single_item, unique_raw), total=len(unique_raw)))

        labels = [item['sentiment'] for item in processed]
        train, temp_data, _, temp_labels = train_test_split(
            processed, labels, test_size=1-train_size, random_state=42, stratify=labels
        )
        val, test = train_test_split(temp_data, test_size=test_size, random_state=42, stratify=temp_labels)

        print(f"Loaded dataset with {len(train)} train, {len(val)} val and {len(test)} test samples.")
        ModelVisualizer.display_dataset_previews(train, val, test)

        # Remove original review text to reduce size
        for dataset in [train, val, test]:
            for item in dataset:
                item.pop('review', None)

        # Save datasets
        os.makedirs(save_dir, exist_ok=True)
        for name, dataset in zip(["train", "val", "test"], [train, val, test]):
            path = os.path.join(save_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Saved {name} set to {path}")

        return train, val, test
    
    @classmethod
    def load_saved_imdb(save_dir="data/IMDb"):
        datasets = {}
        for split in ["train", "val", "test"]:
            path = os.path.join(save_dir, f"{split}.json")
            with open(path, "r", encoding="utf-8") as f:
                datasets[split] = json.load(f)
            print(f"Loaded {len(datasets[split])} samples from {path}")
        return datasets["train"], datasets["val"], datasets["test"]
