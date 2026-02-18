# src/utils/visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from collections import Counter
from tqdm.auto import tqdm
from IPython.display import display
from typing import List, Dict, Any


class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(title, fontsize=14, pad=15)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_certainty_histogram(preds, probs, y_true, lower=0.3, title="Certainty Distribution"):
        """Density histogram showing model certainty regions."""
        upper = 1 - lower
        correctness = np.where(preds == y_true, "Correct", "Incorrect")

        df = pd.DataFrame({'Probability': probs, 'Result': correctness})
        plt.figure(figsize=(12, 8))

        sns.histplot(data=df, x='Probability', hue='Result', bins=40,
                     multiple="stack", palette={'Correct': '#2ecc71', 'Incorrect': '#e74c3c'},
                     kde=True, alpha=0.7)

        plt.axvline(lower, color='black', linestyle='--', alpha=0.5)
        plt.axvline(upper, color='black', linestyle='--', alpha=0.5)
        plt.title(title, fontsize=14)
        plt.xlabel("Probability of Positive Class")
        plt.ylabel("Number of Samples")
        sns.despine()
        plt.show()

    @staticmethod
    def plot_top_concepts(sf, concept_units, top_n=15):
        """Lollipop plot for the most significant concepts (Z-score)."""
        df_pos = sf.logodds_per_class[1].sort_values("zscore", ascending=False).head(top_n)
        df_neg = sf.logodds_per_class[0].sort_values("zscore", ascending=False).head(top_n)

        df_plot = pd.DataFrame({
            "concept_name": [concept_units[c] for c in df_pos["concept"]] +
                            [concept_units[c] for c in df_neg["concept"]],
            "score": list(df_pos["zscore"]) + list(-df_neg["zscore"]),
            "sentiment": ["Positive"] * top_n + ["Negative"] * top_n
        }).sort_values("score")

        plt.figure(figsize=(12, 8))
        colors = {"Positive": "#2ecc71", "Negative": "#e74c3c"}

        plt.hlines(y=df_plot["concept_name"], xmin=0, xmax=df_plot["score"],
                   color=[colors[s] for s in df_plot["sentiment"]], alpha=0.5)

        for sentiment, color in colors.items():
            mask = df_plot["sentiment"] == sentiment
            plt.scatter(df_plot.loc[mask, "score"], df_plot.loc[mask, "concept_name"],
                        color=color, s=100, label=sentiment, edgecolors='white', zorder=3)

        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f"Top {top_n} Discriminative Concepts (Z-score)", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sentiment_wordclouds(train, train_map, unique_units_to_train, sf, top_n=4, max_words=50, n_gram_range=(1,3)):
        """
        Generates a grid of WordClouds for top positive and negative sentiment concepts.
        Each unit (concept) is treated as an indivisible phrase.
        """
        # 1. Grouping units by Concept ID
        cluster_to_units = {}

        for item in tqdm(train, desc="Building WordCloud clusters"):
            from src.utils.loader import DataLoader
            row_units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)

            for u in row_units:
                if u in train_map and u in unique_units_to_train:
                    cid = train_map[u]
                    normalized_unit = u.replace(" ", "_")
                    cluster_to_units.setdefault(cid, []).append(normalized_unit)

        # 2. Choosing concepts based on Z-score
        pos_ids = (
            sf.logodds_per_class[1]
            .sort_values('zscore', ascending=False)
            .head(top_n)['concept']
            .astype(int)
            .tolist()
        )

        neg_ids = (
            sf.logodds_per_class[1]
            .sort_values('zscore', ascending=True)
            .head(top_n)['concept']
            .astype(int)
            .tolist()
        )

        # 3. Preparing the plot grid
        fig, axes = plt.subplots(2, top_n, figsize=(12, 8))

        def draw_wc(ax, cid, colormap, label):
            # Count frequency of phrases in the cluster
            word_freq = Counter(cluster_to_units.get(cid, []))

            wc = WordCloud(
                background_color='white',
                colormap=colormap,
                max_words=max_words,
                width=400,
                height=300,
                regexp=r"\w+"  # allows underscores
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation='bilinear')

            z_score = sf.logodds_per_class[1].loc[cid, 'zscore']
            ax.set_title(
                f"{label} (ID: {cid})\nZ-score: {z_score:.2f}",
                fontsize=12,
                fontweight='bold'
            )
            ax.axis('off')

        # 4. Drawing positive concepts
        for i, cid in enumerate(pos_ids):
            draw_wc(axes[0, i], cid, 'Greens', "POSITIVE")

        # 5. Drawing negative concepts
        for i, cid in enumerate(neg_ids):
            draw_wc(axes[1, i], cid, 'Reds', "NEGATIVE")

        plt.suptitle(
            "Semantic Analysis of Top Sentiment Concepts",
            fontsize=20,
            y=1.02
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_concept_wordcloud(concept_units, title="Global Map of Discovered Concepts"):
        """
        Visualizes representatives of all clusters.
        Each concept is treated as one indivisible phrase.
        """
        # Join phrases using underscores so that WordCloud treats
        # "bad acting" as one token "bad_acting"
        text = " ".join([u.replace(" ", '_') for u in concept_units])

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="tab20",  # Colorful palette for diverse topics
            max_font_size=100,
            random_state=42,
            regexp=r"\w+"
        ).generate(text)

        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(title, fontsize=18, pad=20, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_extreme_errors(data_list: List[Dict], top_n: int = 5):
        """
        Displays reviews where the model was most confident but incorrect.
        """
        errors = [item for item in data_list if item['sentiment'] != item['pred']]
        errors.sort(key=lambda x: x['prob'], reverse=True)

        print(f"\n--- Top {top_n} 'Confident' Positive Errors (Sure it was 1, actually 0) ---")
        display(pd.DataFrame(errors[:top_n])[['clean_review', 'sentiment', 'pred', 'prob']])

        print(f"\n--- Top {top_n} 'Confident' Negative Errors (Sure it was 0, actually 1) ---")
        display(pd.DataFrame(errors[-top_n:])[['clean_review', 'sentiment', 'pred', 'prob']])

    @staticmethod
    def display_uncertain_errors(data_list: List[Dict], top_n: int = 10):
        """
        Displays reviews where the model was most uncertain (probability near 0.5).
        """
        errors = [item for item in data_list if item['sentiment'] != item['pred']]
        errors.sort(key=lambda x: abs(x['prob'] - 0.5))

        print(f"\n--- Top {top_n} Most Uncertain Reviews (Prob near 0.5) ---")
        df_display = pd.DataFrame(errors[:top_n]).copy()
        df_display['uncertainty_score'] = (df_display['prob'] - 0.5).abs()
        display(df_display[['clean_review', 'sentiment', 'pred', 'prob', 'uncertainty_score']])

    @staticmethod
    def display_dataset_previews(train, val, test, n_rows=5):
        """
        Displays a summary and first few rows of each dataset split.
        """
        data_sets = [("TRAIN", train), ("VALIDATION", val), ("TEST", test)]

        for name, ds in data_sets:
            print(f"\n--- {name} SET (Total: {len(ds)} reviews) ---")
            display(pd.DataFrame(ds[:n_rows]))