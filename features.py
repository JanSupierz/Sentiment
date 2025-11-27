import numpy as np
import pandas as pd
from scipy import sparse
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentFeatures:
    """
    Computes class-distinguishing log-odds ratios (with Dirichlet prior)
    for concept groups to identify which concepts are predictive of sentiment.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.concept_list = None
        self.classes = None
        self.logodds_per_class = {}

    # -----------------------------------------------------
    # Build binary concept presence matrix
    # -----------------------------------------------------
    def _build_matrix(self, concept_ids):
        all_concepts = sorted(set(itertools.chain.from_iterable(concept_ids)))
        self.concept_list = all_concepts
        concept_to_idx = {c: i for i, c in enumerate(all_concepts)}

        data, ri, ci = [], [], []
        for r, cs in enumerate(concept_ids):
            for c in cs:
                ri.append(r)
                ci.append(concept_to_idx[c])
                data.append(1)

        X = sparse.csr_matrix(
            (data, (ri, ci)),
            shape=(len(concept_ids), len(all_concepts)),
            dtype=np.int8
        )
        return X

    # -----------------------------------------------------
    # Compute log-odds ratio with prior
    # -----------------------------------------------------
    def _compute_logodds(self, X, sentiments, target_cls):
        y = np.array(sentiments)
        cls_mask = y == target_cls
        not_cls_mask = y != target_cls

        counts_cls = np.asarray(X[cls_mask].sum(axis=0)).ravel()
        counts_not = np.asarray(X[not_cls_mask].sum(axis=0)).ravel()

        n_cls = counts_cls.sum()
        n_not = counts_not.sum()

        alpha = self.alpha
        V = len(counts_cls)

        p_cls = (counts_cls + alpha) / (n_cls + alpha * V)
        p_not = (counts_not + alpha) / (n_not + alpha * V)

        logodds = np.log(p_cls / p_not)
        variance = 1/(counts_cls + alpha) + 1/(counts_not + alpha)
        z_scores = logodds / np.sqrt(variance)

        df = pd.DataFrame({
            "concept": self.concept_list,
            "logodds": logodds,
            "zscore": z_scores
        }).sort_values("zscore", ascending=False)

        return df

    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------
    def fit(self, concept_ids, sentiments):
        sentiments = list(sentiments)
        self.classes = sorted(set(sentiments))

        X = self._build_matrix(concept_ids)

        for cls in self.classes:
            self.logodds_per_class[cls] = self._compute_logodds(
                X, sentiments, cls
            )
        return self

    # -----------------------------------------------------
    # Top concept IDs (not dataframe)
    # -----------------------------------------------------
    def top_n_ids(self, cls, n=20):
        """Return only the concept IDs for the top-N concepts."""
        return self.logodds_per_class[cls]["concept"].head(n).tolist()

    def top_percent_ids(self, cls, pct=5):
        df = self.logodds_per_class[cls]
        n = max(1, int(len(df) * pct / 100))
        return df["concept"].head(n).tolist()

    # -----------------------------------------------------
    # Row extractor (optimized)
    # -----------------------------------------------------
    @staticmethod
    def extract_top_features_row(concept_ids_row, top_set):
        """Optimized for set membership lookups."""
        return [cid for cid in concept_ids_row if cid in top_set]

    # -----------------------------------------------------
    # Apply extraction to a dataframe
    # -----------------------------------------------------
    def add_top_features(self, df, n=20):
        """
        Adds:
            - concept_ids_positive_features
            - concept_ids_negative_features
        to the dataframe.

        positive_cls and negative_cls must exist in sentiments.
        """

        top_pos = set(self.top_n_ids(1, n))
        top_neg = set(self.top_n_ids(0, n))
        top_all = top_pos | top_neg  # combine sets

        df[f"concept_ids_positive_features"] = df["concept_ids"].apply(
            lambda row: self.extract_top_features_row(row, top_pos)
        )
        df[f"concept_ids_negative_features"] = df["concept_ids"].apply(
            lambda row: self.extract_top_features_row(row, top_neg)
        )
        df[f"concept_ids_important_features"] = df["concept_ids"].apply(
            lambda row: self.extract_top_features_row(row, top_all)
        )

        return df

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    def plot_top(self, concept_units, top_n=20):
        # Extract top-N for each class
        df_pos = self.logodds_per_class[1].head(top_n)
        df_neg = self.logodds_per_class[0].head(top_n)

        # Build combined dataframe for plotting
        df_plot = pd.DataFrame({
            "word": [concept_units[c] for c in df_pos["concept"]] +
                    [concept_units[c] for c in df_neg["concept"]],
            "score": list(df_pos["zscore"]) + list(-df_neg["zscore"]),
            "sentiment": ["positive"] * top_n + ["negative"] * top_n
        })

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df_plot,
            x="score",
            y="word",
            hue="sentiment",
            dodge=False,
            palette={"positive": "green", "negative": "red"}
        )

        plt.axvline(0, color="black")
        plt.title(f"Top {top_n}: Positive (green) vs Negative (red)")
        plt.tight_layout()
        plt.show()

