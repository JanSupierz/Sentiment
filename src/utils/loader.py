import re
import unicodedata
import contractions
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

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
        """Lean processing: No n-grams here to save RAM."""
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
        """Standard n-gram generator to be used JIT."""
        tokens = re.findall(r'\w+|[^\w\s]', text)
        min_n, max_n = ngram_range
        units = []
        for n in range(min_n, max_n + 1):
            if n == 1: units.extend(tokens)
            else:
                grams = zip(*[tokens[i:] for i in range(n)])
                units.extend([" ".join(g) for g in grams])
        return units

    @classmethod
    def load_imdb(cls, n_jobs=4):
        print("Loading IMDb reviews...")
        ds = tfds.load('imdb_reviews', split='train+test', as_supervised=True)
        raw_data = list(tfds.as_numpy(ds))
        
        seen = set()
        unique_raw = [ (t, l) for t, l in raw_data if not (t in seen or seen.add(t)) ]
        
        print(f"Cleaning {len(unique_raw)} reviews...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            processed = list(tqdm(executor.map(cls.process_single_item, unique_raw), total=len(unique_raw)))

        labels = [item['sentiment'] for item in processed]
        train, temp_data, _, temp_labels = train_test_split(
            processed, labels, test_size=0.40, random_state=42, stratify=labels
        )
        val, test = train_test_split(temp_data, test_size=0.50, random_state=42, stratify=temp_labels)
        
        return train, val, test