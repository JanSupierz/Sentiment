# src/features/sentiment.py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import itertools
from typing import List, Union, Optional


class SentimentFeatures:
    """
    Compute log‑odds and Z‑scores for each feature (concept or unit).
    Can accept both list‑of‑lists (backward compatibility) and CSR matrix.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.logodds_per_class = {}   # dict {0: df, 1: df}
        self.concept_list = None     # list of feature IDs (columns)
        self._feature_to_idx = None

    def fit(self,
            X: Union[List[List[int]], csr_matrix],
            y: List[int],
            feature_names: Optional[List[int]] = None):
        """
        Fit sentiment statistics.
        If X is CSR matrix: shape (n_docs, n_features)
        If X is list of lists: each inner list contains feature IDs (may have duplicates)
        feature_names: only needed if X is list-of-lists (to define column order)
        """
        if isinstance(X, csr_matrix):
            self._fit_sparse(X, y)
        else:
            self._fit_list(X, y, feature_names)
        return self

    def _fit_sparse(self, X: csr_matrix, y: List[int]):
        """Fast path: directly use CSR matrix."""
        n_docs, n_features = X.shape
        self.concept_list = list(range(n_features))   # assume column indices are feature IDs
        self._feature_to_idx = {f: i for i, f in enumerate(self.concept_list)}

        y_arr = np.array(y)
        for cls in [0, 1]:
            mask = (y_arr == cls)
            counts_cls = X[mask].sum(axis=0).A1
            counts_not = X[~mask].sum(axis=0).A1

            p_cls = (counts_cls + self.alpha) / (counts_cls.sum() + self.alpha * n_features)
            p_not = (counts_not + self.alpha) / (counts_not.sum() + self.alpha * n_features)

            logodds = np.log(p_cls / p_not)
            z_scores = logodds / np.sqrt(1/(counts_cls + self.alpha) + 1/(counts_not + self.alpha))

            self.logodds_per_class[cls] = pd.DataFrame({
                "concept": self.concept_list,
                "zscore": z_scores
            })

    def _fit_list(self, X: List[List[int]], y: List[int], feature_names: Optional[List[int]] = None):
        """Original list‑of‑lists implementation."""
        unique_concepts = sorted(set(itertools.chain.from_iterable(X)))
        if feature_names is not None:
            # Use provided order (important for consistency)
            unique_concepts = feature_names
        self.concept_list = unique_concepts
        self._feature_to_idx = {c: i for i, c in enumerate(unique_concepts)}

        # Build sparse matrix
        rows, cols, data = [], [], []
        for doc_id, concepts in enumerate(X):
            for c in concepts:
                if c in self._feature_to_idx:
                    rows.append(doc_id)
                    cols.append(self._feature_to_idx[c])
                    data.append(1)   # each occurrence counts
        X_mat = coo_matrix((data, (rows, cols)),
                           shape=(len(X), len(unique_concepts))).tocsr()
        self._fit_sparse(X_mat, y)

    def filter_by_zscore(self, threshold: float) -> set:
        """
        Return set of concept IDs whose Z‑score > threshold for either class.
        """
        pos_set = set(self.logodds_per_class[1][self.logodds_per_class[1]['zscore'] > threshold]['concept'])
        neg_set = set(self.logodds_per_class[0][self.logodds_per_class[0]['zscore'] > threshold]['concept'])
        return pos_set | neg_set