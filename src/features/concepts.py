# src/features/concepts.py
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional, Union
from sklearn.feature_selection import SelectKBest, f_regression


class ConceptExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        self.selected_dims = None   # for supervised dimension reduction
        self._embeddings_cache = None   # optional: store precomputed embeddings

    def set_precomputed_embeddings(self, embeddings: np.ndarray, units: List[str]):
        """
        Inject precomputed embeddings (e.g., from cache) to avoid recomputation.
        Embeddings must be normalized and in the same order as `units`.
        """
        self._embeddings_cache = (units, embeddings)

    def train_concepts(self,
                       units: List[str],
                       sentiment_map: Optional[Dict[str, float]] = None,
                       retention_percentile: int = 10,
                       n_clusters: int = 5000,
                       batch_size: int = 128,
                       printing: bool = True) -> Dict[str, Any]:
        """
        Cluster units into concepts.
        If embeddings have been precomputed (via set_precomputed_embeddings), they are used.
        """
        unique_units = sorted(list(dict.fromkeys(units)))
        n_clusters = min(n_clusters, len(unique_units))

        # --- Embeddings ---
        if self._embeddings_cache is not None:
            cached_units, cached_emb = self._embeddings_cache
            if cached_units == unique_units:
                if printing:
                    print("Using precomputed embeddings from cache.")
                embeddings_np = cached_emb
            else:
                raise ValueError("Cached embeddings do not match provided units.")
        else:
            if printing:
                print(f"Generating embeddings for {len(unique_units)} unique units...")
            with torch.inference_mode():
                embeddings = self.model.encode(
                    unique_units,
                    batch_size=batch_size,
                    show_progress_bar=printing,
                    convert_to_tensor=True,
                    device=self.device
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings_np = embeddings.cpu().float().numpy().astype(np.float32)
                del embeddings
                torch.cuda.empty_cache()

        # --- Supervised Dimension Reduction ---
        if sentiment_map is not None:
            if printing:
                print("Performing Supervised Dimension Reduction (Sentiment Focus)...")
            y = np.array([sentiment_map.get(u, 0.5) for u in unique_units], dtype=np.float32)

            k = int(embeddings_np.shape[1] * (retention_percentile / 100))
            k = max(k, 1)

            selector = SelectKBest(f_regression, k=k)
            selector.fit(embeddings_np, y)

            self.selected_dims = selector.get_support(indices=True)
            embeddings_np = embeddings_np[:, self.selected_dims]

            if printing:
                print(f"Reduced embedding dimensions from {selector.n_features_in_} to {k}.")

        # --- FAISS Clustering ---
        if printing:
            print(f"FAISS Clustering (n={n_clusters})...")
        d = embeddings_np.shape[1]

        # Use GPU if available
        gpu_res = None
        if faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True

        kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=printing, gpu=gpu_res is not None)
        kmeans.train(embeddings_np)
        cluster_centers = kmeans.centroids

        # Assign each unit to nearest centroid
        index = faiss.IndexFlatL2(d)
        if gpu_res:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(embeddings_np)
        _, labels = index.search(cluster_centers, 1)   # find closest unit to each centroid
        rep_indices = labels.flatten()
        concept_units = [unique_units[idx] for idx in rep_indices]

        # Actually assign units to clusters (using centroids)
        index.reset()
        index.add(cluster_centers)
        _, unit_labels = index.search(embeddings_np, 1)
        unit_labels = unit_labels.flatten()

        return {
            "cluster_centers": torch.tensor(cluster_centers),
            "concept_units": concept_units,
            "unit_to_cluster": {u: int(l) for u, l in zip(unique_units, unit_labels)},
            "n_concepts": n_clusters
        }

    def map_units_to_clusters(self,
                              units: List[str],
                              cluster_centers: torch.Tensor,
                              batch_size: int = 128,
                              printing: bool = True) -> Dict[str, int]:
        """
        Map new units to existing clusters (centroids).
        Uses cosine similarity (via L2 on normalized vectors) with threshold.
        """
        if not units:
            return {}
        unique_units = sorted(list(dict.fromkeys(units)))

        with torch.inference_mode():
            embeddings = self.model.encode(
                unique_units,
                batch_size=batch_size,
                show_progress_bar=printing,
                convert_to_tensor=True,
                device=self.device
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().float().numpy()
            del embeddings
            torch.cuda.empty_cache()

        if self.selected_dims is not None:
            embeddings_np = embeddings_np[:, self.selected_dims]

        centers_np = cluster_centers.cpu().numpy().astype(np.float32)
        d = centers_np.shape[1]

        # Build index on centroids
        index = faiss.IndexFlatL2(d)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(centers_np)

        distances, labels = index.search(embeddings_np, 1)
        # Convert L2^2 to cosine similarity (since vectors are normalized)
        similarities = 1 - distances.flatten() / 2
        SIM_THRESHOLD = 0.6

        result = {}
        for u, l, s in zip(unique_units, labels.flatten(), similarities):
            if s >= SIM_THRESHOLD:
                result[u] = int(l)
        return result