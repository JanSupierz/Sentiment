#!/usr/bin/env python3
"""
Visualizes the specific words or concepts that drive sentiment.
Includes individual Top 5 Positive and Top 5 Negative Cluster visualization.
"""
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from src.utils.paths import get_figure_path, DATA_DIR

sns.set_theme(style="whitegrid")

def load_artifacts(ngram_range, n_concepts, weight):
    """Loads vocabulary, concepts, and sentiment statistics."""
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    
    # 1. Load Vocabulary
    vocab_path = DATA_DIR / "vocab" / f"vocab_{key}.pkl"
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found: {vocab_path}")
        return None, None, None

    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # 2. Load Concept Mapping (if applicable)
    concept_data = None
    if n_concepts > 0:
        filename = f"concepts_{key}_k{n_concepts}_w{int(weight)}.pkl"
        concept_path = DATA_DIR / "concepts" / filename
        
        if not concept_path.exists():
            print(f"❌ Concept file not found: {concept_path}")
            return None, None, None

        with open(concept_path, 'rb') as f:
            concept_data = pickle.load(f)
        
        # Generate feature names for concepts (since we don't have 'concept_units')
        feature_names = [f"Concept_{i}" for i in range(n_concepts)]
        repr_key = f"{key}_k{n_concepts}_w{int(weight)}"
    else:
        feature_names = vocab_data['vocab']
        repr_key = f"{key}_raw"

    # 3. Load Statistics (Z-Scores)
    stats_path = DATA_DIR / "stats" / f"stats_{repr_key}.pkl"
    if not stats_path.exists():
        print(f"❌ Stats not found: {stats_path}")
        return None, None, None

    with open(stats_path, 'rb') as f:
        logodds = pickle.load(f)
        
    return feature_names, logodds, concept_data

def plot_sentiment_importance(feature_names, logodds, title, output_path):
    """Generates the lollipop chart for top discriminative features."""
    top_pos = logodds[1].sort_values("zscore", ascending=False).head(15)
    top_neg = logodds[0].sort_values("zscore", ascending=False).head(15)
    
    pos_words = [feature_names[i] for i in top_pos['concept']]
    neg_words = [feature_names[i] for i in top_neg['concept']]
    
    df_plot = pd.DataFrame({
        "Feature": pos_words + neg_words,
        "Z-Score": list(top_pos['zscore']) + list(-top_neg['zscore']), 
        "Sentiment": ["Positive"] * 15 + ["Negative"] * 15
    }).sort_values("Z-Score")
    
    plt.figure(figsize=(10, 8))
    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c"}
    
    plt.hlines(y=df_plot["Feature"], xmin=0, xmax=df_plot["Z-Score"], 
               color=[colors[s] for s in df_plot["Sentiment"]], alpha=0.5, linewidth=2)
    
    for sentiment, color in colors.items():
        mask = df_plot["Sentiment"] == sentiment
        plt.scatter(df_plot.loc[mask, "Z-Score"], df_plot.loc[mask, "Feature"], 
                    color=color, s=80, label=sentiment, edgecolors='white', zorder=3)
        
    plt.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel("Sentiment Strength (Z-Score)")
    plt.ylabel("Feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved importance plot to {output_path}")

def plot_top_cluster_grid(concept_data, logodds, class_id, label, weight, top_n=5, n_concepts=None):
    """Generates a grid of word clouds for the top N clusters of a specific sentiment."""
    if not concept_data:
        return

    unit_to_cluster = concept_data['unit_to_cluster']
    cluster_contents = {}
    for word, cid in unit_to_cluster.items():
        cluster_contents.setdefault(cid, []).append(word.replace(" ", "_"))

    top_clusters = logodds[class_id].sort_values("zscore", ascending=False).head(top_n)
    
    fig, axes = plt.subplots(1, top_n, figsize=(22, 5))
    fig.suptitle(f"Top {top_n} {label} Semantic Clusters (Weight: {weight})", fontsize=18)

    for i, (_, row) in enumerate(top_clusters.iterrows()):
        cid = int(row['concept'])
        zscore = row['zscore']
        words = cluster_contents.get(cid, ["empty"])
        
        from collections import Counter
        word_freq = Counter(words)
        
        wc = WordCloud(width=400, height=400, background_color="white", 
                       colormap="Greens" if class_id == 1 else "Reds").generate_from_frequencies(word_freq)
        
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"Cluster {cid}\nZ: {zscore:.2f}", fontsize=12)
        axes[i].axis("off")

    out_path = get_figure_path("vocabulary", f"top_{top_n}_{label}_w{int(weight)}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved {label} cluster grid to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmin", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=3)
    parser.add_argument("--n_concepts", type=int, default=5000)
    parser.add_argument("--sentiment_weight", type=float, default=0.0)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    
    print(f"--- Visualizing: Concepts={args.n_concepts}, Weight={args.sentiment_weight} ---")
    
    feature_names, logodds, concept_data = load_artifacts(
        (args.nmin, args.nmax), 
        args.n_concepts, 
        args.sentiment_weight
    )
    
    if feature_names is None:
        return

    run_name = f"ngram_{args.nmin}_{args.nmax}_k{args.n_concepts}_w{int(args.sentiment_weight)}"
    title = f"Weight: {args.sentiment_weight} | K: {args.n_concepts}"
    
    # 1. Lollipop Importance Plot
    importance_path = get_figure_path("vocabulary", f"{run_name}_importance.png")
    plot_sentiment_importance(feature_names, logodds, title, importance_path)

    # 2. Individual Cluster Grids (if using concepts)
    if args.n_concepts > 0:
        plot_top_cluster_grid(concept_data, logodds, 1, "Positive", args.sentiment_weight, args.top_n)
        plot_top_cluster_grid(concept_data, logodds, 0, "Negative", args.sentiment_weight, args.top_n)

if __name__ == "__main__":
    main()