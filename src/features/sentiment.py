import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import itertools
from typing import List

class SentimentFeatures:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.logodds_per_class = {}
        self.concept_list = None

    def fit(self, concept_ids_list: List[List[int]], sentiments: List[int]):
        unique_concepts = sorted(set(itertools.chain.from_iterable(concept_ids_list)))
        self.concept_list = unique_concepts
        c_to_idx = {c: i for i, c in enumerate(unique_concepts)}
        
        rows, cols = zip(*[(r, c_to_idx[c]) for r, concepts in enumerate(concept_ids_list) for c in concepts])
        data = np.ones(len(rows), dtype=np.float32)
        X = coo_matrix((data, (rows, cols)), shape=(len(concept_ids_list), len(unique_concepts))).tocsr()
        y = np.array(sentiments)

        for cls in [0, 1]:
            mask = (y == cls)
            counts_cls = X[mask].sum(axis=0).A1
            counts_not = X[~mask].sum(axis=0).A1
            
            p_cls = (counts_cls + self.alpha) / (counts_cls.sum() + self.alpha * len(unique_concepts))
            p_not = (counts_not + self.alpha) / (counts_not.sum() + self.alpha * len(unique_concepts))
            
            logodds = np.log(p_cls / p_not)
            z_scores = logodds / np.sqrt(1/(counts_cls + self.alpha) + 1/(counts_not + self.alpha))
            self.logodds_per_class[cls] = pd.DataFrame({"concept": unique_concepts, "zscore": z_scores})
        return self