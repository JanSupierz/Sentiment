import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List


class SentimentFeatures:
    """
    Compute log-odds and Z-scores for each feature (concept or unit)
    using a pre-computed sparse CSR matrix.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.logodds_per_class = {}   # dict {0: df, 1: df}
        self.concept_list = None      # list of feature IDs (columns)
        self._feature_to_idx = None

    def fit(self, X: csr_matrix, y: List[int]):
        n_docs, n_features = X.shape
        self.concept_list = list(range(n_features))   # column indices are IDs
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

        return self

    def filter_by_zscore(self, threshold: float) -> set:
        """
        Return set of concept IDs whose Z-score > threshold for either class.
        """
        pos_set = set(self.logodds_per_class[1][self.logodds_per_class[1]['zscore'] > threshold]['concept'])
        neg_set = set(self.logodds_per_class[0][self.logodds_per_class[0]['zscore'] > threshold]['concept'])
        return pos_set | neg_set