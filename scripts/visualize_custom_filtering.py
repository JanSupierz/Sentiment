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
        help="The dataframe column containing the tokens to visualize (e.g., tokens_lower, tokens_filtered)"
    )
    args = parser.parse_args()
    target_col = args.column

    print("Loading data for visualization...")
    train_df = pd.read_parquet("data/preprocessed/train.parquet")

    if target_col not in train_df.columns:
        raise ValueError(f"Column '{target_col}' not found in the dataset. Available columns: {list(train_df.columns)}")

    y_train = train_df["sentiment"].values

    print(f"Vectorizing text for '{target_col}' (max_df=0.7)...")
    X_train, _, count_vect = build_count_matrix(train_df[target_col], None, max_df=0.7)
    vocab = count_vect.get_feature_names_out()

    print("Calculating Z-scores on the Custom matrix...")
    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    df_scores = sf.logodds_per_class[1].copy()
    df_scores['word'] = [vocab[i] for i in df_scores['concept']]

    z_threshold = 2

    # --- SPLIT INTO TRASHED AND KEPT ---
    trashed_df = df_scores[df_scores['zscore'].abs() <= z_threshold].copy()
    trashed_df['abs_z'] = trashed_df['zscore'].abs()
    trashed_df = trashed_df.sort_values('abs_z', ascending=True)

    kept_pos = df_scores[df_scores['zscore'] > z_threshold].sort_values('zscore', ascending=False)
    kept_neg = df_scores[df_scores['zscore'] < -z_threshold].sort_values('zscore', ascending=True)

    # --- GENERATE VISUALIZATIONS ---
    print("\nGenerating WordClouds (with punctuation preserved)...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Sentiment Analysis: {target_col}", fontsize=20, fontweight='bold', y=1.05)

    # Panel 1: Trashed (Greys) - using \S+ to keep punctuation
    trashed_text = " ".join([w.replace(" ", "_") for w in trashed_df['word'].head(150)])
    wc_trash = WordCloud(background_color='white', colormap='Greys', width=400, height=400, regexp=r"\S+").generate(trashed_text)
    axes[0].imshow(wc_trash, interpolation='bilinear')
    axes[0].set_title(f"Trashed Words (Neutral Noise)\nTotal Discarded: {len(trashed_df)}", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Kept Positive (Greens)
    pos_text = " ".join([w.replace(" ", "_") for w in kept_pos['word'].head(150)])
    wc_pos = WordCloud(background_color='white', colormap='Greens', width=400, height=400, regexp=r"\S+").generate(pos_text)
    axes[1].imshow(wc_pos, interpolation='bilinear')
    axes[1].set_title(f"Kept Positive Words\nTotal Kept: {len(kept_pos)}", fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Kept Negative (Reds)
    neg_text = " ".join([w.replace(" ", "_") for w in kept_neg['word'].head(150)])
    wc_neg = WordCloud(background_color='white', colormap='Reds', width=400, height=400, regexp=r"\S+").generate(neg_text)
    axes[2].imshow(wc_neg, interpolation='bilinear')
    axes[2].set_title(f"Kept Negative Words\nTotal Kept: {len(kept_neg)}", fontsize=16, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    out_path = FIGURES_DIR / "analysis" / f"{target_col}_wordclouds.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison image to {out_path}")


if __name__ == "__main__":
    main()