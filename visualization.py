import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        labels = sorted(np.unique(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_certainty_analysis(preds, probs, y_true, lower: float, title="Certainty Analysis"):
        import matplotlib.gridspec as gridspec
        upper = 1 - lower

        # Categorize certainty
        categories = np.empty(len(probs), dtype=object)
        categories[probs < lower] = "Certain Negative"
        categories[probs > upper] = "Certain Positive"
        categories[(probs >= lower) & (probs <= upper)] = "Uncertain"

        # Correctness
        correctness = np.where(preds == y_true, "Correct", "Incorrect")

        # Heatmap data
        table = pd.crosstab(categories, correctness, normalize="index")
        # Optional: reorder rows for consistent heatmap
        row_order = ["Certain Positive", "Certain Negative", "Uncertain"]
        table = table.reindex(row_order)

        # Bar plot data
        counts = pd.Series(categories).value_counts(normalize=True) * 100  # percentage
        counts = counts.reindex(row_order)  # enforce fixed order

        # Define fixed colors
        color_map = {
            "Uncertain": "gray",
            "Certain Negative": "red",
            "Certain Positive": "green"
        }

        # Create combined figure
        plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Heatmap subplot
        ax0 = plt.subplot(gs[0])
        sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues", ax=ax0)
        ax0.set_title(f"{title} - Certainty vs Correctness")
        ax0.set_xlabel("Correctness")
        ax0.set_ylabel("Certainty class")

        # Bar chart subplot
        ax1 = plt.subplot(gs[1])
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            palette=[color_map[c] for c in counts.index],
            order=counts.index,
            legend=False,
            ax=ax1
        )

        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Percentage (%)")
        ax1.set_title(f"{title} - Certainty Distribution")
        for i, v in enumerate(counts.values):
            ax1.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.show()
