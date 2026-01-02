import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

class ConceptExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)

    def train_concepts(self, units: List[str], n_clusters: int = 5000, batch_size: int = 128) -> Dict[str, Any]:
        unique_units = sorted(list(dict.fromkeys(units)))
        n_clusters = min(n_clusters, len(unique_units))

        print(f"Generowanie embeddingów dla {len(unique_units)} unikalnych jednostek...")
        with torch.inference_mode():
            # SBERT ma wbudowany pasek postępu (show_progress_bar=True)
            embeddings = self.model.encode(
                unique_units, 
                batch_size=batch_size, 
                show_progress_bar=True, 
                convert_to_tensor=True,
                device=self.device
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().float().numpy().astype(np.float32)
            del embeddings
            torch.cuda.empty_cache()

        print(f"Klastrowanie FAISS (n={n_clusters})...")
        d = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(d)
        kmeans = faiss.Clustering(d, n_clusters)
        kmeans.niter = 20
        kmeans.verbose = True
        kmeans.train(embeddings_np, index)

        cluster_centers = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, d)
        _, labels = index.search(embeddings_np, 1)
        labels = labels.flatten()

        # Szybki wybór reprezentantów
        centroid_index = faiss.IndexFlatL2(d)
        centroid_index.add(embeddings_np)
        _, rep_indices = centroid_index.search(cluster_centers, 1)
        concept_units = [unique_units[idx[0]] for idx in rep_indices]

        return {
            "cluster_centers": torch.tensor(cluster_centers),
            "concept_units": concept_units,
            "unit_to_cluster": {u: int(l) for u, l in zip(unique_units, labels)},
            "n_concepts": n_clusters
        }

    def map_units_to_clusters(self, units: List[str], cluster_centers: torch.Tensor, batch_size: int = 128) -> Dict[str, int]:
        if not units: return {}
        unique_units = sorted(list(dict.fromkeys(units)))
        
        with torch.inference_mode():
            embeddings = self.model.encode(
                unique_units, 
                batch_size=batch_size, 
                show_progress_bar=True, 
                convert_to_tensor=True,
                device=self.device
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().numpy()
            del embeddings
            torch.cuda.empty_cache()

        centers_np = cluster_centers.cpu().numpy().astype(np.float32)
        index = faiss.IndexFlatL2(centers_np.shape[1])
        index.add(centers_np)
        distances, labels = index.search(embeddings_np, 1)

        # Filtr podobieństwa 0.5 (dist^2 <= 1.0)
        return {u: int(l) for u, d, l in zip(unique_units, distances.flatten(), labels.flatten()) if d <= 1.0}