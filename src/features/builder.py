# src/features/builder.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
import torch
from sentence_transformers import SentenceTransformer
import faiss
faiss.verbose = False
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.loader import DataLoader
from src.features.sentiment import SentimentFeatures
from src.features.concept_remap import remap_sparse_matrix
from src.utils.paths import DATA_DIR

_SENTENCE_MODEL = None
_ANALYZER = None

def _get_sentence_model(model_name: str = 'all-MiniLM-L6-v2'):
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _SENTENCE_MODEL = SentenceTransformer(model_name).to(device)
    return _SENTENCE_MODEL

def _get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER

def build_ngram_index(cfg, ngram_range, splits=None):
    """
    Build vocabulary and count matrices for specified splits.
    If splits is None, builds all three (train, val, test).
    """
    if splits is None:
        splits = ['train', 'val', 'test']
        
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    vocab_path = DATA_DIR / "vocab" / f"vocab_{key}.pkl"
    cache_dir = DATA_DIR / "cache_matrices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. JIT VOCABULARY BUILDING ---
    if not vocab_path.exists():
        print(f"Vocab missing for {key}. Building from train split...")
        try:
            train_df = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
        except FileNotFoundError:
            raise ValueError("Train split (train.parquet) must be available to build vocabulary.")

        min_freq = cfg['data']['min_term_freq']
        max_df_ratio = cfg['data']['max_df_ratio']

        df_by_label = defaultdict(Counter)
        for _, row in train_df.iterrows():
            units = set(DataLoader.get_ngrams(row["clean_bow"], ngram_range))
            df_by_label[row["sentiment"]].update(units)

        n_docs_label = Counter(train_df["sentiment"])
        stop_sets = []
        for label in [0, 1]:
            thresh = n_docs_label[label] * max_df_ratio
            stop_sets.append({u for u, cnt in df_by_label[label].items() if cnt > thresh})
        stop_units = set.intersection(*stop_sets) if stop_sets else set()

        tf_counts = Counter()
        for _, row in train_df.iterrows():
            units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
            tf_counts.update([u for u in units if u not in stop_units])

        vocab = sorted([u for u, cnt in tf_counts.items() if cnt >= min_freq])
        unit_to_id = {u: i for i, u in enumerate(vocab)}

        DATA_DIR.joinpath("vocab").mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "wb") as f:
            pickle.dump({"vocab": vocab, "unit_to_id": unit_to_id, "stop_units": list(stop_units)}, f)

    # --- 2. JIT SPLIT MATRIX BUILDING ---
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
        
    unit_to_id = vocab_data['unit_to_id']
    stop_units = set(vocab_data['stop_units'])
    vocab_len = len(vocab_data['vocab'])

    for sp in splits:
        cache_file = cache_dir / f"X_{sp}_{key}.npz"
        if not cache_file.exists():
            print(f"Cache missing for {sp} split. Building matrix...")
            df = pd.read_parquet(DATA_DIR / "preprocessed" / f"{sp}.parquet")
            
            rows, cols, data = [], [], []
            for doc_id, (_, row) in enumerate(df.iterrows()):
                units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
                for u in units:
                    if u in stop_units:
                        continue
                    uid = unit_to_id.get(u)
                    if uid is not None:
                        rows.append(doc_id)
                        cols.append(uid)
                        data.append(1)
                        
            X = csr_matrix((data, (rows, cols)), shape=(len(df), vocab_len), dtype=np.float32)
            save_npz(cache_file, X)


def compute_and_cache_embeddings(cfg, ngram_range):
    """Compute SBERT embeddings + VADER sentiment scores for all units (always needed if clustering)."""
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    cache_path = DATA_DIR / "cache_matrices" / f"emb_{key}.npz"
    if cache_path.exists():
        return
        
    with open(DATA_DIR / "vocab" / f"vocab_{key}.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    units = vocab_data['vocab']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = cfg['bert']['sentence_model']
    model = SentenceTransformer(model_name).to(device)
    embeddings = model.encode(units, batch_size=256, convert_to_tensor=True,
                              device=device, show_progress_bar=False)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = np.array([analyzer.polarity_scores(u)['compound'] for u in units],
                                dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(cache_path, embeddings=embeddings, sentiment_scores=sentiment_scores)


def run_extraction_logic(ngram_range, n_concepts, sentiment_weight):
    """Cluster units into concepts (if n_concepts>0) and save centroids."""
    if n_concepts == 0:
        return
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    out_dir = DATA_DIR / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"concepts_{key}_k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if out_path.exists():
        return

    with open(DATA_DIR / "vocab" / f"vocab_{key}.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    units = vocab_data['vocab']

    data = np.load(DATA_DIR / "cache_matrices" / f"emb_{key}.npz")
    aug_embeddings = np.hstack([
        data['embeddings'],
        data['sentiment_scores'] * float(sentiment_weight)
    ]).astype(np.float32)
    aug_embeddings = aug_embeddings / np.linalg.norm(aug_embeddings, axis=1, keepdims=True)
    
    actual_k = min(n_concepts, len(units))
    kmeans = faiss.Kmeans(aug_embeddings.shape[1], actual_k, niter=20,
                          verbose=False, gpu=torch.cuda.is_available())
    kmeans.train(aug_embeddings)
    _, labels = kmeans.index.search(aug_embeddings, 1)

    centroids = kmeans.centroids
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Save both mapping and centroids
    with open(out_path, "wb") as f:
        pickle.dump({
            'unit_to_cluster': dict(zip(units, labels.flatten().tolist())),
            'centroids': centroids,
            'sentiment_weight': sentiment_weight,
            'ngram_range': ngram_range,
            'n_concepts': actual_k
        }, f)
        

def run_stats_logic(cfg, ngram_range, n_concepts, sentiment_weight):
    """Compute log‑odds statistics for the representation (requires train matrix)."""
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    repr_key = f"{key}_k{n_concepts}_w{int(sentiment_weight)}" if n_concepts > 0 else f"{key}_raw"
    out_dir = DATA_DIR / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stats_{repr_key}.pkl"
    if out_path.exists():
        return

    # Ensure train matrix exists for this ngram_range
    train_cache = DATA_DIR / "cache_matrices" / f"X_train_{key}.npz"
    if not train_cache.exists():
        build_ngram_index(cfg, ngram_range, splits=['train'])

    X_train = load_npz(train_cache)
    y_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")['sentiment'].values

    if n_concepts > 0:
        with open(DATA_DIR / "concepts" / f"concepts_{key}_k{n_concepts}_w{int(sentiment_weight)}.pkl", "rb") as f:
            c_data = pickle.load(f)
        with open(DATA_DIR / "vocab" / f"vocab_{key}.pkl", "rb") as f:
            v_data = pickle.load(f)
        id_to_cluster = {v_data['unit_to_id'][u]: cid for u, cid in c_data['unit_to_cluster'].items()}
        X_train = remap_sparse_matrix(X_train, id_to_cluster, n_concepts)

    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())
    with open(out_path, "wb") as f:
        pickle.dump(sf.logodds_per_class, f)


def ensure_representation(cfg, ngram_range, n_concepts, sentiment_weight, splits=None):
    """
    Ensure that all required files for a given representation exist for the specified splits.
    If splits is None, builds all three splits.
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    build_ngram_index(cfg, ngram_range, splits)
    compute_and_cache_embeddings(cfg, ngram_range)
    if n_concepts > 0:
        run_extraction_logic(ngram_range, n_concepts, sentiment_weight)
    run_stats_logic(cfg, ngram_range, n_concepts, sentiment_weight)


def map_unseen_units(unseen_units, concept_data_path, model_name='all-MiniLM-L6-v2', batch_size=256):
    """
    Map a list of unit strings (unseen during training) to concept IDs using saved centroids.
    Returns dict {unit: concept_id}.
    """
    # Load centroids and metadata
    with open(concept_data_path, 'rb') as f:
        data = pickle.load(f)
    centroids = data['centroids']
    sentiment_weight = data['sentiment_weight']

    # Get models
    model = _get_sentence_model(model_name)
    analyzer = _get_analyzer()

    # Compute embeddings for unseen units (batched on GPU)
    embeddings = model.encode(unseen_units, batch_size=batch_size, convert_to_tensor=True,
                              device=model.device, show_progress_bar=False)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    sent_scores = np.array([analyzer.polarity_scores(u)['compound'] for u in unseen_units], dtype=np.float32).reshape(-1, 1)
    embeddings = np.hstack([embeddings, sent_scores * sentiment_weight])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build FAISS index on centroids (done once per call, not per word)
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids.astype(np.float32))
    distances, labels = index.search(embeddings.astype(np.float32), 1)

    # Similarity threshold (cosine from L2 on normalized vectors)
    SIM_THRESHOLD = 0.6
    similarities = 1 - distances.flatten() / 2
    mapping = {}
    for u, l, sim in zip(unseen_units, labels.flatten(), similarities):
        if sim >= SIM_THRESHOLD:
            mapping[u] = int(l)
    return mapping


def build_concept_matrices(cfg, ngram_range, n_concepts, sentiment_weight, splits=None):
    """
    Build and save concept-level sparse matrices for the requested splits.
    If splits is None, builds for all three (train, val, test).
    Matrices are saved in data/concept_matrices/.
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    concept_file = DATA_DIR / "concepts" / f"concepts_{key}_k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not concept_file.exists():
        raise FileNotFoundError(f"Concept file not found: {concept_file}. Run run_extraction_logic first.")

    # Load concept data (centroids + training mapping)
    with open(concept_file, 'rb') as f:
        cdata = pickle.load(f)
    unit_to_cluster_train = cdata['unit_to_cluster']   # mapping for training units only
    centroids = cdata['centroids']
    actual_n_concepts = centroids.shape[0]

    # Load vocabulary (unit_to_id)
    vocab_file = DATA_DIR / "vocab" / f"vocab_{key}.pkl"
    with open(vocab_file, 'rb') as f:
        vdata = pickle.load(f)
    unit_to_id = vdata['unit_to_id']
    id_to_unit = {idx: unit for unit, idx in unit_to_id.items()}

    # Cache for mapping of unseen units (per configuration, reused across splits)
    mapping_cache = {}

    # Preload sentence transformer model name and frequency threshold from cfg
    model_name = cfg['bert']['sentence_model']
    min_freq = cfg['data']['min_term_freq']

    for split in splits:
        out_dir = DATA_DIR / "concept_matrices"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{split}_{key}_k{n_concepts}_w{int(sentiment_weight)}.npz"
        
        if out_file.exists():
            print(f"Concept matrix for {split} already exists: {out_file}")
            continue
            
        print(f"Building concept matrix for {split} (k={n_concepts}, w={sentiment_weight})...")
        df = pd.read_parquet(DATA_DIR / "preprocessed" / f"{split}.parquet")

        if split == 'train':
            # Use the unit matrix and remap via training mapping
            unit_mat = load_npz(DATA_DIR / "cache_matrices" / f"X_train_{key}.npz")
            id_to_cluster = {}
            for uid in range(unit_mat.shape[1]):
                unit = id_to_unit.get(uid)
                if unit is not None and unit in unit_to_cluster_train:
                    id_to_cluster[uid] = unit_to_cluster_train[unit]
                    
            if not id_to_cluster:
                X_concept = csr_matrix((len(df), 0), dtype=np.float32)
            else:
                valid_uids = sorted(id_to_cluster.keys())
                remap = np.array([id_to_cluster[uid] for uid in valid_uids])
                X_unit_subset = unit_mat[:, valid_uids]
                coo = X_unit_subset.tocoo()
                rows, cols, data = coo.row, remap[coo.col], coo.data
                df_agg = pd.DataFrame({'row': rows, 'col': cols, 'data': data}).groupby(['row', 'col'], as_index=False).sum()
                X_concept = csr_matrix((df_agg['data'].values,
                                        (df_agg['row'].values, df_agg['col'].values)),
                                       shape=(len(df), actual_n_concepts))
        else:
            # ===== OPTIMIZED 3-PASS SYSTEM FOR VAL/TEST =====
            
            # Pass 1: Count frequencies of all units in this split
            split_counts = Counter()
            for _, row in df.iterrows():
                units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
                split_counts.update(units)
                
            # Filter for unseen units that meet the minimum frequency threshold
            unseen_set = set()
            for u, count in split_counts.items():
                if count >= min_freq:
                    uid = unit_to_id.get(u)
                    # If unit is completely new, OR in vocab but wasn't clustered
                    if uid is None or id_to_unit.get(uid) not in unit_to_cluster_train:
                        if u not in mapping_cache:
                            unseen_set.add(u)

            # Pass 2: Batch-map ONLY the frequent unseen units
            if unseen_set:
                unseen_list = list(unseen_set)
                print(f"Batch mapping {len(unseen_list)} frequent new units for {split} (dropped rare units)...")
                new_mappings = map_unseen_units(unseen_list, concept_file, model_name=model_name)
                
                for u in unseen_list:
                    mapping_cache[u] = new_mappings.get(u, -1)

            # Pass 3: Build the sparse matrix
            rows, cols, data = [], [], []
            for doc_id, (_, row) in enumerate(df.iterrows()):
                units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
                for u in units:
                    uid = unit_to_id.get(u)
                    
                    # If it's a known training unit, we always use it
                    if uid is not None and id_to_unit.get(uid) in unit_to_cluster_train:
                        cid = unit_to_cluster_train[id_to_unit[uid]]
                    else:
                        # Otherwise, check the cache (returns -1 if it was rare/ignored or below similarity threshold)
                        cid = mapping_cache.get(u, -1)

                    if cid != -1:
                        rows.append(doc_id)
                        cols.append(cid)
                        data.append(1)

            if len(rows) == 0:
                X_concept = csr_matrix((len(df), 0), dtype=np.float32)
            else:
                X_concept = coo_matrix((data, (rows, cols)),
                                       shape=(len(df), actual_n_concepts)).tocsr()

        save_npz(out_file, X_concept)
        print(f"Saved concept matrix to {out_file}")


def load_representation(cfg, ngram_range, n_concepts, sentiment_weight, z_threshold, split):
    """
    Load the sparse count matrix for a given split after ensuring it exists.
    Applies Z‑score filtering if z_threshold > 0.
    Returns (X_sparse, y_array).
    """
    key = f"{ngram_range[0]}_{ngram_range[1]}"
    y = pd.read_parquet(DATA_DIR / "preprocessed" / f"{split}.parquet")['sentiment'].values

    if n_concepts == 0:
        # Original unit-level representation
        build_ngram_index(cfg, ngram_range, splits=[split])
        X = load_npz(DATA_DIR / "cache_matrices" / f"X_{split}_{key}.npz")
        n_features = X.shape[1]
        repr_key = f"{key}_raw"
    else:
        # Concept-level representation – use precomputed matrices
        concept_dir = DATA_DIR / "concept_matrices"
        concept_file = concept_dir / f"{split}_{key}_k{n_concepts}_w{int(sentiment_weight)}.npz"
        if not concept_file.exists():
            # Build concept matrices for all splits if missing (this will be done only once)
            build_concept_matrices(cfg, ngram_range, n_concepts, sentiment_weight, splits=[split])
        X = load_npz(concept_file)
        n_features = X.shape[1]
        repr_key = f"{key}_k{n_concepts}_w{int(sentiment_weight)}"

    # Z-score filtering
    if z_threshold > 0:
        stats_path = DATA_DIR / "stats" / f"stats_{repr_key}.pkl"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        important_pos = set(stats[1][stats[1]['zscore'] > z_threshold]['concept'])
        important_neg = set(stats[0][stats[0]['zscore'] > z_threshold]['concept'])
        important = important_pos | important_neg
        mask = np.array([i in important for i in range(n_features)])
        X = X[:, mask]

    return X, y