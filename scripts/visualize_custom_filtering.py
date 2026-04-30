import argparse
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.features.sentiment import SentimentFeatures
from src.utils.paths import FIGURES_DIR
from src.features.vectorizer import build_count_matrix


def main():
    parser = argparse.ArgumentParser(description="Generate WordClouds for specific token columns.")
    parser.add_argument(
        "--column",
        type=str,
        default="tokens_lower",
        help="The dataframe column containing the tokens to visualize"
    )
    args = parser.parse_args()
    target_col = args.column

    print("Loading data for visualization...")
    train_df = pd.read_parquet("data/preprocessed/train.parquet")

    if target_col not in train_df.columns:
        raise ValueError(f"Column '{target_col}' not found in the dataset. Available columns: {list(train_df.columns)}")

    y_train = train_df["sentiment"].values

    print(f"Vectorizing text for '{target_col}'")
    X_train, _, count_vect = build_count_matrix(train_df[target_col], None)
    vocab = count_vect.get_feature_names_out()

    print("Calculating Z-scores on the Custom matrix...")
    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    df_scores = sf.logodds_per_class[1].copy()
    df_scores['word'] = [vocab[i] for i in df_scores['concept']]

    z_threshold = 2

    trashed_df = df_scores[df_scores['zscore'].abs() <= z_threshold].copy()
    trashed_df['abs_z'] = trashed_df['zscore'].abs()
    trashed_df = trashed_df.sort_values('abs_z', ascending=True)

    kept_pos = df_scores[df_scores['zscore'] > z_threshold].sort_values('zscore', ascending=False)
    kept_neg = df_scores[df_scores['zscore'] < -z_threshold].sort_values('zscore', ascending=True)

    print("\nGenerating WordClouds individually (DPI=300)...")

    out_dir = FIGURES_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_wordcloud(word_scores_dict, caption_text, colormap, filename):
        """Save a word cloud as a PDF using frequencies with no built‑in title, and print the caption."""
        # Replace spaces with underscores to keep n-grams grouped as single elements
        formatted_dict = {str(k).replace(" ", "_"): float(v) for k, v in word_scores_dict.items()}

        wc = WordCloud(
            background_color='white',
            colormap=colormap,
            width=800,
            height=800,
            regexp=r"\S+"
        ).generate_from_frequencies(formatted_dict)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.savefig(out_dir / filename, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)

        print(f"Saved: {out_dir / filename}  →  caption: \"{caption_text}\"")

    # Generate dictionaries for the top 150 words
    neutral_top = trashed_df.head(150)
    neutral_dict = dict(zip(neutral_top['word'], neutral_top['abs_z']))

    pos_top = kept_pos.head(150)
    pos_dict = dict(zip(pos_top['word'], pos_top['zscore']))

    neg_top = kept_neg.head(150)
    neg_dict = dict(zip(neg_top['word'], neg_top['zscore'].abs()))

    # Save the WordClouds
    save_wordcloud(
        word_scores_dict=pos_dict,
        caption_text=f"Extracted positive words (total kept: {len(kept_pos)})",
        colormap='Greens',
        filename=f"{target_col}_positive_wordcloud.pdf"
    )

    save_wordcloud(
        word_scores_dict=neutral_dict,
        caption_text=f"Extracted neutral words (total discarded: {len(trashed_df)})",
        colormap='Greys',
        filename=f"{target_col}_neutral_wordcloud.pdf"
    )

    save_wordcloud(
        word_scores_dict=neg_dict,
        caption_text=f"Extracted negative words (total kept: {len(kept_neg)})",
        colormap='Reds',
        filename=f"{target_col}_negative_wordcloud.pdf"
    )

    print("\nAll three wordclouds saved successfully (DPI=300).")


if __name__ == "__main__":
    main()