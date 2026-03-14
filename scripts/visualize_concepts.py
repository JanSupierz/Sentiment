"""
Visualize concepts (clusters) for two sentiment weights, e.g., 0 and 10.
Loads concept mapping and stats, displays top positive/negative clusters.
"""
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.utils.paths import DATA_DIR, FIGURES_DIR


def load_concept_data(token_col, n_concepts, sentiment_weight):
    """Load concept mapping and stats for a given configuration."""
    # Concept mapping (unit_to_cluster)
    concept_path = DATA_DIR / "concepts" / token_col / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not concept_path.exists():
        raise FileNotFoundError(f"Concept file not found: {concept_path}")
    with open(concept_path, "rb") as f:
        concept_data = pickle.load(f)
    unit_to_cluster = concept_data['unit_to_cluster']
    n_concepts_actual = concept_data['n_concepts']

    # Stats (logodds per class) – saved by builder.compute_concept_z_indices
    stats_path = DATA_DIR / "stats" / token_col / "concepts" / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)   # dict {0: df, 1: df}

    return unit_to_cluster, stats, n_concepts_actual


def plot_top_concepts(unit_to_cluster, stats, sentiment_weight, top_n=10, save_path=None):
    """Vertically compact lollipop plot with clear class separation."""
    # Load and process data
    pos_df = stats[1].sort_values('zscore', ascending=False).head(top_n).copy()
    neg_df = stats[0].sort_values('zscore', ascending=False).head(top_n).copy()

    concept_repr = {}
    for word, cid in unit_to_cluster.items():
        if cid not in concept_repr:
            concept_repr[cid] = word.replace(' ', '_')

    pos_df['label'] = [concept_repr.get(c, f"c{c}") for c in pos_df['concept']]
    neg_df['label'] = [concept_repr.get(c, f"c{c}") for c in neg_df['concept']]

    pos_df['plot_z'] = pos_df['zscore']
    neg_df['plot_z'] = -neg_df['zscore']

    # Sorting for Vertical Layout
    pos_df = pos_df.sort_values('plot_z', ascending=True)
    neg_df = neg_df.sort_values('plot_z', ascending=True)

    pos_df['Sentiment'] = 'Positive'
    neg_df['Sentiment'] = 'Negative'

    # Insert a separator row
    spacer = pd.DataFrame({'label': ['───'], 'plot_z': [0], 'Sentiment': ['Neutral']})
    plot_data = pd.concat([neg_df, spacer, pos_df], ignore_index=True)

    # Compact Canvas Setup
    plt.figure(figsize=(9, 10), facecolor='#F8F9FA') 
    ax = plt.gca()

    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}

    for i, row in plot_data.iterrows():
        color = colors[row['Sentiment']]
        if row['Sentiment'] == 'Neutral':
            plt.text(0, i, "• • •", ha='center', va='center', color=color, fontweight='bold')
            continue

        plt.hlines(y=i, xmin=0, xmax=row['plot_z'], color=color, alpha=0.5, linewidth=1.5)
        plt.scatter(row['plot_z'], i, color=color, s=60, edgecolors='white', zorder=3)

    # Tighten Labels and Spines
    ax.set_xlim(-65, 65)
    plt.yticks(range(len(plot_data)), plot_data['label'], fontsize=9, fontweight='bold')
    plt.xticks(fontsize=5)

    plt.axvline(0, color='black', linewidth=0.8, alpha=0.3)

    # Compact Title
    plt.title(f"Concept Discrimination (Weight={sentiment_weight})", 
              fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Z-score Intensity", fontsize=10, fontweight='bold')

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved compact lollipop plot to {save_path}")


def plot_cluster_wordclouds(unit_to_cluster, stats, sentiment_weight, class_id=1, top_n=5, max_words=50, save_path=None):
    """Word clouds for top N clusters of a given class (positive=1, negative=0)."""
    # Group words by cluster
    cluster_to_words = {}
    for word, cid in unit_to_cluster.items():
        cluster_to_words.setdefault(cid, []).append(word.replace(' ', '_'))

    class_df = stats[class_id].sort_values('zscore', ascending=False).head(top_n)
    top_clusters = class_df['concept'].tolist()

    fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 5))
    if top_n == 1:
        axes = [axes]
    label = "Positive" if class_id == 1 else "Negative"
    fig.suptitle(f"Top {top_n} {label} Clusters (weight={sentiment_weight})", fontsize=16)

    for i, cid in enumerate(top_clusters):
        words = cluster_to_words.get(cid, ["empty"])
        freq = {w: 1 for w in words}   # equal weight; could use counts if available
        wc = WordCloud(width=400, height=400, background_color='white',
                       colormap='Greens' if class_id == 1 else 'Reds',
                       max_words=max_words).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"Cluster {cid}\nZ={class_df.iloc[i]['zscore']:.2f}", fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved word cloud grid to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_col', default='tokens_lower', help='Token column name')
    parser.add_argument('--n_concepts', type=int, default=5000, help='Number of concepts')
    parser.add_argument('--weights', nargs=2, type=float, default=[0.0, 10.0],
                        help='Two sentiment weights to compare')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top concepts to display')
    args = parser.parse_args()

    for w in args.weights:
        print(f"\n=== Processing weight = {w} ===")
        unit_to_cluster, stats, n_actual = load_concept_data(args.token_col, args.n_concepts, w)
        print(f"Loaded {len(unit_to_cluster)} units mapped to {n_actual} concepts.")

        # Lollipop plot
        lollipop_path = FIGURES_DIR / "concepts" / f"lollipop_{args.token_col}_k{args.n_concepts}_w{int(w)}.png"
        lollipop_path.parent.mkdir(parents=True, exist_ok=True)
        plot_top_concepts(unit_to_cluster, stats, w, top_n=args.top_n, save_path=lollipop_path)

        # Word clouds for positive
        pos_wc_path = FIGURES_DIR / "concepts" / f"pos_wc_{args.token_col}_k{args.n_concepts}_w{int(w)}.png"
        plot_cluster_wordclouds(unit_to_cluster, stats, w, class_id=1, top_n=5,
                                save_path=pos_wc_path)

        # Word clouds for negative
        neg_wc_path = FIGURES_DIR / "concepts" / f"neg_wc_{args.token_col}_k{args.n_concepts}_w{int(w)}.png"
        plot_cluster_wordclouds(unit_to_cluster, stats, w, class_id=0, top_n=5,
                                save_path=neg_wc_path)


if __name__ == "__main__":
    main()