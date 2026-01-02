import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
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

    def plot_top(self, concept_units, top_n=15):
            """
            Tworzy elegancki wykres Lollipop dla najbardziej dyskryminatywnych konceptów.
            """
            # 1. Przygotowanie danych (Z-score dla klasy 1 jest dodatni, dla klasy 0 ujemny)
            df_pos = self.logodds_per_class[1].sort_values("zscore", ascending=False).head(top_n).copy()
            df_neg = self.logodds_per_class[0].sort_values("zscore", ascending=False).head(top_n).copy()
            
            # Łączymy w jeden DataFrame
            df_plot = pd.DataFrame({
                "concept_name": [concept_units[c] for c in df_pos["concept"]] + [concept_units[c] for c in df_neg["concept"]],
                "score": list(df_pos["zscore"]) + list(-df_neg["zscore"]),
                "sentiment": ["Pozytywny"] * top_n + ["Negatywny"] * top_n
            })
            
            # Sortujemy, żeby wykres "płynął" od góry do dołu
            df_plot = df_plot.sort_values("score", ascending=True)

            # 2. Stylizacja
            plt.figure(figsize=(12, 10))
            sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})
            
            colors = {"Pozytywny": "#2ecc71", "Negatywny": "#e74c3c"}
            
            # Rysujemy linie "lizaków"
            plt.hlines(y=df_plot["concept_name"], xmin=0, xmax=df_plot["score"], 
                    color=[colors[s] for s in df_plot["sentiment"]], alpha=0.6, linewidth=2)
            
            # Rysujemy kropki "lizaków"
            for sentiment, color in colors.items():
                mask = df_plot["sentiment"] == sentiment
                plt.scatter(df_plot.loc[mask, "score"], df_plot.loc[mask, "concept_name"], 
                            color=color, s=100, label=sentiment, edgecolors='white', zorder=3)

            # 3. Dodatki estetyczne
            plt.axvline(0, color='black', linewidth=0.8, alpha=0.5) # Linia zero
            
            # Ograniczamy długość tekstu konceptu, żeby nie rozwaliło wykresu
            plt.gca().set_yticklabels([text[:60] + '...' if len(text) > 60 else text for text in df_plot["concept_name"]])
            
            plt.title(f"Top {top_n} Konceptów Dyskryminatywnych\n(Standaryzowany współczynnik Log-Odds)", 
                    fontsize=16, pad=20, fontweight='bold')
            plt.xlabel("Wartość Z-score (Siła wpływu na klasyfikację)", fontsize=12)
            plt.legend(title="Sentyment", loc="lower right", frameon=True)
            
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.show()