# src/features/concept_remap.py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def remap_sparse_matrix(X_unit: csr_matrix, unit_to_concept: dict, n_concepts: int = None):
    """
    Transform CSR matrix with columns = unit IDs into CSR matrix with columns = concept IDs.

    Parameters
    ----------
    X_unit : csr_matrix, shape (n_docs, n_units)
    unit_to_concept : dict
        Mapping from unit ID (int) to concept ID (int). Units not in mapping are dropped.
    n_concepts : int, optional
        Total number of concepts (must be >= max(concept ID)+1). If None, inferred.

    Returns
    -------
    X_concept : csr_matrix
    """
    X_unit = X_unit.tocoo()
    rows = X_unit.row
    cols = X_unit.col
    data = X_unit.data

    # Map column indices
    new_cols = np.array([unit_to_concept.get(c, -1) for c in cols], dtype=np.int32)
    mask = new_cols != -1
    rows = rows[mask]
    new_cols = new_cols[mask]
    data = data[mask]

    if len(rows) == 0:
        # No concepts mapped – return empty matrix
        return csr_matrix((X_unit.shape[0], 0), dtype=data.dtype)

    if n_concepts is None:
        n_concepts = max(new_cols) + 1

    X_concept = coo_matrix((data, (rows, new_cols)), shape=(X_unit.shape[0], n_concepts))
    return X_concept.tocsr()