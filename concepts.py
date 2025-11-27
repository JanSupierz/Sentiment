import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, List, Any
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import faiss

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def train_concepts(
        self, 
        units: List[str], 
        n_clusters: int = 500, 
        batch_size: int = 1024,
        use_gpu: bool = True
    ) -> Dict[str, Any]:

        # Remove duplicates while preserving order
        unique_units = list(dict.fromkeys(units))
        logger.info(f"Found {len(unique_units)} unique units.")
        
        if len(unique_units) < n_clusters:
            logger.warning(
                f"Reducing n_clusters from {n_clusters} to {len(unique_units)} "
                "(number of unique units)"
            )
            n_clusters = len(unique_units)

        # Compute embeddings
        logger.info("Computing embeddings...")
        embeddings = self.model.encode(
            unique_units, 
            convert_to_tensor=True, 
            batch_size=batch_size, 
            show_progress_bar=True
        )

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        # FAISS clustering
        d = embeddings_np.shape[1]
        logger.info(f"Clustering {len(embeddings_np)} embeddings into {n_clusters} concepts using FAISS...")

        if use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0  # GPU 0
            index = faiss.GpuIndexFlatL2(res, d, flat_config)
        else:
            index = faiss.IndexFlatL2(d)

        kmeans = faiss.Clustering(d, n_clusters)
        kmeans.max_iter = 100
        kmeans.nredo = 1
        kmeans.verbose = True

        # Train KMeans
        kmeans.train(embeddings_np, index)

        # Extract cluster centers
        cluster_centers = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, d)

        # Assign points to clusters
        _, labels = index.search(embeddings_np, 1)
        labels = labels.flatten()

        # Find concept representatives
        concept_units = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                distances = np.linalg.norm(embeddings_np - cluster_centers[cluster_id], axis=1)
                rep_idx = np.argmin(distances)
            else:
                cluster_embeddings = embeddings_np[mask]
                indices = np.where(mask)[0]
                distances = np.linalg.norm(cluster_embeddings - cluster_centers[cluster_id], axis=1)
                rep_idx = indices[np.argmin(distances)]
            concept_units.append(unique_units[rep_idx])

        # Map each unit to its cluster
        unit_to_cluster = {unit: int(label) for unit, label in zip(unique_units, labels)}

        logger.info("Concept training completed!")
        return {
            "cluster_centers": torch.tensor(cluster_centers),
            "concept_units": concept_units,
            "unit_to_cluster": unit_to_cluster,
            "n_concepts": n_clusters
        }
    
    def _find_cluster_representatives(
        self, 
        units: List[str], 
        embeddings: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray
    ) -> List[str]:

        representatives = []
        for cluster_id in range(len(centers)):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                distances = np.linalg.norm(embeddings - centers[cluster_id], axis=1)
                rep_idx = np.argmin(distances)
            else:
                cluster_embeddings = embeddings[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                distances = np.linalg.norm(cluster_embeddings - centers[cluster_id], axis=1)
                rep_idx = cluster_indices[np.argmin(distances)]
            representatives.append(units[rep_idx])
        return representatives

    def map_units_to_clusters(
        self, 
        units: List[str], 
        cluster_centers: torch.Tensor, 
        batch_size: int = 1024
    ) -> Dict[str, int]:

        if not units:
            return {}

        unique_units = list(dict.fromkeys(units))
        logger.info(f"Mapping {len(unique_units)} units to nearest clusters...")

        centers_np = cluster_centers.cpu().numpy()

        embeddings = self.model.encode(
            unique_units, 
            convert_to_tensor=True, 
            batch_size=batch_size, 
            show_progress_bar=True
        )

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()

        # Cosine similarity
        similarities = embeddings_np @ centers_np.T
        cluster_assignments = np.argmax(similarities, axis=1)

        return {unit: int(cluster_id) for unit, cluster_id in zip(unique_units, cluster_assignments)}


def visualize_concept_wordcloud(concept_units: List[str], max_words: int = 200):
    if not concept_units:
        logger.warning("No concept units to visualize")
        return
    
    safe_units = [u.replace(" ", '_') for u in concept_units]
    text = " ".join(safe_units)

    wc = WordCloud(
        width=1200, 
        height=600, 
        background_color="white", 
        max_words=max_words, 
        collocations=False
    ).generate(text)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Concept Representative Word Cloud", fontsize=20)
    plt.tight_layout()
    plt.show()


def conceptualize_units(unit_list: List[str], unit_to_cluster_map: Dict[str, int]) -> List[int]:
    concept_ids = []
    for u in unit_list:
        if u in unit_to_cluster_map:
            concept_ids.append(unit_to_cluster_map[u])
        else:
            logger.warning(f"Unit not in cluster map: {u}")
    return concept_ids


def conceptualize_df(df, column: str, unit_to_cluster_map: Dict[str, int], new_col: str = "concept_ids"):
    def convert(units):
        if isinstance(units, str):
            units = units.split()
        if not isinstance(units, list):
            logger.error(f"Unexpected unit format: {units}")
            return []
        return conceptualize_units(units, unit_to_cluster_map)

    df[new_col] = df[column].apply(convert)
