import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
import torch
from sentence_transformers import SentenceTransformer
import faiss
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.paths import DATA_DIR
from src.features.vectorizer import build_count_matrix
from src.features.sentiment import SentimentFeatures

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


def build_unit_matrices(token_col, ngram_range=(1,3), min_df=10, max_df=0.7, force=False):
    # Ensure ngram_range is a tuple (YAML loads it as a list)
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)

    cache_dir = DATA_DIR / "cache_matrices" / token_col
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train.npz"
    val_path = cache_dir / "val.npz"
    test_path = cache_dir / "test.npz"
    vocab_path = cache_dir / "vocab.pkl"

    print(f"Building unit matrices for '{token_col}' ...")
    train_df = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")

    # Prepare token lists (each is a list of tokens)
    train_tokens = train_df[token_col].tolist()
    val_tokens = val_df[token_col].tolist()
    test_tokens = test_df[token_col].tolist()

    # Build matrices for train and val simultaneously
    X_train, X_val, vectorizer = build_count_matrix(
        train_tokens, val_tokens,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range
    )

    # Build test matrix using the same vectorizer
    test_reviews = [" ".join(tokens) for tokens in test_tokens]
    X_test = vectorizer.transform(test_reviews)

    # Save matrices
    save_npz(train_path, X_train)
    save_npz(val_path, X_val)
    save_npz(test_path, X_test)

    # Save vocabulary (feature names)
    with open(vocab_path, "wb") as f:
        pickle.dump(vectorizer.get_feature_names_out(), f)

    print(f"Unit matrices for '{token_col}' saved.")
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'vocab': vocab_path,
    }


def compute_unit_z_indices(token_col, z_scores, force=False):
    cache_dir = DATA_DIR / "cache_matrices" / token_col / "unit_z_indices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load train matrix
    train_path = DATA_DIR / "cache_matrices" / token_col / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Train matrix for '{token_col}' not found.")
    X_train = load_npz(train_path)
    y_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")['sentiment'].values

    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    for z in z_scores:
        keep_set = sf.filter_by_zscore(z)
        keep_indices = sorted(keep_set)
        out_path = cache_dir / f"z_{z}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(keep_indices, f)
        print(f"   Unit Z={z}: kept {len(keep_indices)} features.")


def compute_embeddings(token_col, ngram_range=(1,3), model_name='all-MiniLM-L6-v2'):
    # Ensure ngram_range is a tuple
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)

    cache_dir = DATA_DIR / "cache_matrices" / token_col
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npz"
    if emb_path.exists():
        return

    vocab_path = cache_dir / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary for '{token_col}' not found.")
    with open(vocab_path, "rb") as f:
        units = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name).to(device)
    embeddings = model.encode(units, batch_size=256, convert_to_tensor=True,
                              device=device, show_progress_bar=False)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = np.array([analyzer.polarity_scores(u)['compound'] for u in units],
                                dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(emb_path, embeddings=embeddings, sentiment_scores=sentiment_scores)
    print(f"Embeddings for '{token_col}' saved.")


def extract_concepts(token_col, n_concepts, sentiment_weight, force=False):
    out_dir = DATA_DIR / "concepts" / token_col
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"

    # Load units and embeddings
    vocab_path = DATA_DIR / "cache_matrices" / token_col / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary for '{token_col}' not found.")
    with open(vocab_path, "rb") as f:
        units = pickle.load(f)

    emb_path = DATA_DIR / "cache_matrices" / token_col / "embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings for '{token_col}' not found.")
    data = np.load(emb_path)

    # Augment embeddings with weighted sentiment
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

    with open(out_path, "wb") as f:
        pickle.dump({
            'unit_to_cluster': dict(zip(units, labels.flatten().tolist())),
            'centroids': centroids,
            'sentiment_weight': sentiment_weight,
            'n_concepts': actual_k
        }, f)
    print(f"Concepts for '{token_col}' k={n_concepts} w={sentiment_weight} saved.")


def build_concept_matrices(token_col, n_concepts, sentiment_weight, force=False):
    concept_dir = DATA_DIR / "cache_matrices" / token_col / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    concept_key = f"k{n_concepts}_w{int(sentiment_weight)}"
    out_train = concept_dir / f"{concept_key}_train.npz"
    out_val = concept_dir / f"{concept_key}_val.npz"
    out_test = concept_dir / f"{concept_key}_test.npz"

    # Load unit-to-concept mapping
    map_path = DATA_DIR / "concepts" / token_col / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not map_path.exists():
        raise FileNotFoundError(f"Concept mapping for '{token_col}' {concept_key} not found.")
    with open(map_path, "rb") as f:
        map_data = pickle.load(f)
    unit_to_cluster = map_data['unit_to_cluster']
    n_concepts_actual = map_data['n_concepts']

    # Build mapping from unit index to concept index
    # Load vocabulary to get unit index
    vocab_path = DATA_DIR / "cache_matrices" / token_col / "vocab.pkl"
    with open(vocab_path, "rb") as f:
        units = pickle.load(f)
    unit_to_idx = {u: i for i, u in enumerate(units)}
    idx_to_cluster = {}
    for u, cid in unit_to_cluster.items():
        idx = unit_to_idx.get(u)
        if idx is not None:
            idx_to_cluster[idx] = cid

    # Helper to remap a unit matrix to concept matrix
    def remap_unit_matrix(unit_mat_path, out_path, n_docs):
        unit_mat = load_npz(unit_mat_path)
        # Convert to COO for easy manipulation
        coo = unit_mat.tocoo()
        rows = coo.row
        cols = coo.col
        data = coo.data
        # Map unit columns to concept columns
        new_cols = np.array([idx_to_cluster.get(c, -1) for c in cols], dtype=np.int32)
        mask = new_cols != -1
        rows = rows[mask]
        new_cols = new_cols[mask]
        data = data[mask]
        if len(rows) == 0:
            concept_mat = csr_matrix((n_docs, 0), dtype=np.float32)
        else:
            concept_mat = coo_matrix((data, (rows, new_cols)),
                                     shape=(n_docs, n_concepts_actual)).tocsr()
        save_npz(out_path, concept_mat)

    # Get number of documents per split
    n_train = len(pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet"))
    n_val   = len(pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet"))
    n_test  = len(pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet"))

    unit_train = DATA_DIR / "cache_matrices" / token_col / "train.npz"
    unit_val   = DATA_DIR / "cache_matrices" / token_col / "val.npz"
    unit_test  = DATA_DIR / "cache_matrices" / token_col / "test.npz"

    remap_unit_matrix(unit_train, out_train, n_train)
    remap_unit_matrix(unit_val,   out_val,   n_val)
    remap_unit_matrix(unit_test,  out_test,  n_test)

    print(f"Concept matrices for '{token_col}' {concept_key} saved.")


def compute_concept_z_indices(token_col, n_concepts, sentiment_weight, z_scores, force=False):
    concept_dir = DATA_DIR / "cache_matrices" / token_col / "concepts"
    concept_key = f"k{n_concepts}_w{int(sentiment_weight)}"
    indices_dir = concept_dir / "z_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    stats_dir = DATA_DIR / "stats" / token_col / "concepts"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{concept_key}.pkl"

    # Check if everything already exists
    indices_exist = all((indices_dir / f"{concept_key}_z{z}.pkl").exists() for z in z_scores)
    stats_exist = stats_path.exists()
    if not force and indices_exist and stats_exist:
        print(f"Concept Z-score indices and stats for '{token_col}' {concept_key} already exist.")
        return

    # Load concept train matrix
    train_path = concept_dir / f"{concept_key}_train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Concept train matrix for '{token_col}' {concept_key} not found.")
    X_train = load_npz(train_path)
    y_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")['sentiment'].values

    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    # Save keep indices for each z
    for z in z_scores:
        keep_set = sf.filter_by_zscore(z)
        keep_indices = sorted(keep_set)
        out_path = indices_dir / f"{concept_key}_z{z}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(keep_indices, f)
        print(f"   Concept Z={z}: kept {len(keep_indices)} features.")

    # Save full stats
    with open(stats_path, "wb") as f:
        pickle.dump(sf.logodds_per_class, f)
    print(f"   Saved full stats to {stats_path}")


def load_representation(token_col, n_concepts, sentiment_weight, z_threshold, split, ngram_range=(1,3)):
    """Loads feature matrix and labels, dynamically building dependencies if they don't exist."""

    # Ensure root Unit Matrices exist
    unit_path = DATA_DIR / "cache_matrices" / token_col / f"{split}.npz"
    if not unit_path.exists():
        print(f"Unit matrix for '{token_col}' split {split} not found. Triggering build...")
        build_unit_matrices(token_col, ngram_range=ngram_range)

    y = pd.read_parquet(DATA_DIR / "preprocessed" / f"{split}.parquet")['sentiment'].values

    if n_concepts == 0:
        # --- UNIT LEVEL ---
        X = load_npz(unit_path)

        # Ensure Unit Z-score indices exist
        indices_dir = DATA_DIR / "cache_matrices" / token_col / "unit_z_indices"
        indices_path = indices_dir / f"z_{z_threshold}.pkl"

        if not indices_path.exists():
            print(f"Unit Z-score indices for z={z_threshold} not found. Triggering compute...")
            compute_unit_z_indices(token_col, z_scores=[z_threshold])

        with open(indices_path, "rb") as f:
            keep_indices = pickle.load(f)
        X = X[:, keep_indices]

    else:
        # --- CONCEPT LEVEL ---
        concept_dir = DATA_DIR / "cache_matrices" / token_col / "concepts"
        concept_key = f"k{n_concepts}_w{int(sentiment_weight)}"
        mat_path = concept_dir / f"{concept_key}_{split}.npz"

        if not mat_path.exists():
            print(f"Concept matrix for '{token_col}' {concept_key} split {split} not found. Checking dependencies...")

            # Embeddings Dependency
            emb_path = DATA_DIR / "cache_matrices" / token_col / "embeddings.npz"
            if not emb_path.exists():
                print("Embeddings not found. Computing...")
                compute_embeddings(token_col, ngram_range=ngram_range)

            # Concept Extraction Dependency
            map_path = DATA_DIR / "concepts" / token_col / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
            if not map_path.exists():
                print("Concepts not found. Extracting...")
                extract_concepts(token_col, n_concepts, sentiment_weight)

            # Build Concept Matrices
            print("Building concept matrices...")
            build_concept_matrices(token_col, n_concepts, sentiment_weight)

        X = load_npz(mat_path)

        # Ensure Concept Z-score indices exist
        indices_dir = concept_dir / "z_indices"
        indices_path = indices_dir / f"{concept_key}_z{z_threshold}.pkl"

        if not indices_path.exists():
            print(f"Concept Z-score indices for {concept_key} z={z_threshold} not found. Triggering compute...")
            compute_concept_z_indices(token_col, n_concepts, sentiment_weight, z_scores=[z_threshold])

        with open(indices_path, "rb") as f:
            keep_indices = pickle.load(f)
        X = X[:, keep_indices]

    return X, y