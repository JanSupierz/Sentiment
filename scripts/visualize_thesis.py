import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.paths import RESULTS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def main():
    input_csv = RESULTS_DIR / "thesis" / "cascade_simulation_results.csv"
    out_plot = FIGURES_DIR / "thesis" / "cascade_comparison_f1.png"

    if not input_csv.exists():
        print(f"Error: {input_csv} not found. Please run the simulation script first.")
        return

    # Load the simulation results
    df = pd.read_csv(input_csv)

    # Extract the metadata for the title
    threshold = df['Threshold'].iloc[0]
    delegation_rate = df['Delegation_Rate'].iloc[0]
    metrics_df = df.drop(columns=['Threshold', 'Delegation_Rate'])

    # Reshape the data for seaborn
    plot_data = metrics_df.T.reset_index()
    plot_data.columns = ['Model', 'F1_Score']

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_data, x='Model', y='F1_Score', palette='viridis', hue='Model', legend=False)

    # Add the exact F1 scores on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.4f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontweight='bold')

    plt.title(f"Cascade System Comparison (F1 Score)\nThreshold: {threshold:.2f} | Data Delegated: {delegation_rate:.1%}", 
              fontweight='bold', pad=15)
    plt.xlabel("System Architecture", fontweight='bold', labelpad=10)
    plt.ylabel("Weighted F1 Score", fontweight='bold', labelpad=10)

    min_score = plot_data['F1_Score'].min()
    max_score = plot_data['F1_Score'].max()
    plt.ylim(max(0, min_score - 0.05), min(1.0, max_score + 0.05))

    # Clean up the X-axis labels
    labels = [label.get_text().split('. ', 1)[-1] if '. ' in label.get_text() else label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    print(f"Visualization saved successfully to: {out_plot}")


if __name__ == "__main__":
    main()