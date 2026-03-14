import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path


def plot_ablation_study(base_dir="results/val", run_names=None):
    
    if run_names is None:
        run_names = ["run_on_preprocessing", "run_custom", "run_basic"]

    # Dynamically construct paths
    base_path = Path(base_dir)
    search_dirs = [base_path / run / "classification_reports" for run in run_names]
    valid_dirs = [d for d in search_dirs if d.exists()]

    if not valid_dirs:
        print(f"Could not find any classification_reports in: {base_path}")
        return

    # Search valid directories for a specific filename
    def find_file(filename):
        for directory in valid_dirs:
            file_path = directory / filename
            if file_path.exists():
                return file_path
        return None

    # Configuration
    token_columns = [
        'tokens_cased', 'tokens_lower', 'tokens_letters',
        'tokens_filtered', 'tokens_stemmed', 'tokens_lemmatized'
    ]

    baseline_col = "text_expanded"
    baseline_casing = "lower"
    baseline_reps = ["BoW", "TF-IDF"]
    representations = ["BoW", "TF-IDF", "CustomTfidf"]
    models = ["linear_svm", "logreg"]
    z_threshold = 2

    records = []
    baselines = {"Linear SVM": {}, "Logistic Regression": {}}

    # Extract Data
    for model in models:
        model_label = "Linear SVM" if model == "linear_svm" else "Logistic Regression"

        # Baselines
        for rep in baseline_reps:
            base_filename = f"{baseline_col}_{baseline_casing}_{rep}_{model}.csv"
            base_file = find_file(base_filename)
            if base_file:
                base_df = pd.read_csv(base_file, index_col=0)
                baselines[model_label][rep] = base_df.loc["macro avg", "f1-score"]

        # Ablation study variants
        for token_col in token_columns:
            for rep in representations:
                if rep == "CustomTfidf":
                    filename = f"{token_col}_{rep}_Z{z_threshold}_{model}.csv"
                else:
                    filename = f"{token_col}_{rep}_{model}.csv"

                filepath = find_file(filename)
                if filepath:
                    df = pd.read_csv(filepath, index_col=0)
                    records.append({
                        "Token Processing": token_col,
                        "Representation": rep,
                        "Model": model_label,
                        "Macro F1-Score": df.loc["macro avg", "f1-score"]
                    })

    if not records:
        print("No matching experiment CSVs found in the provided run folders!")
        return

    # Build DataFrame
    results_df = pd.DataFrame(records).sort_values(by="Macro F1-Score", ascending=False)

    # Create Plot
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=results_df, kind="bar",
        x="Macro F1-Score", y="Token Processing",
        hue="Representation", hue_order=["CustomTfidf", "TF-IDF", "BoW"],
        col="Model", height=8, aspect=1.2,
        palette="viridis", alpha=0.9
    )
    g.set_titles("{col_name}")

    base_styles = {
        "TF-IDF": {"color": "coral", "ls": "--"},
        "BoW": {"color": "navy", "ls": ":"}
    }

    # Add Baselines
    for ax in g.axes.flat:
        title = ax.get_title()
        if title in baselines:
            for rep, score in baselines[title].items():
                style = base_styles[rep]
                ax.axvline(x=score, color=style["color"], linestyle=style["ls"], linewidth=2.5)

    for ax in g.axes.flat:
        best_width = 0
        best_patch = None

        # Iterate through every bar drawn on this specific subplot
        for patch in ax.patches:
            width = patch.get_width() 
            if width > best_width:
                best_width = width
                best_patch = patch

        # Annotate the winning bar
        if best_patch:
            ax.text(
                best_width + 0.0005,
                best_patch.get_y() + best_patch.get_height() / 2,
                f"★ {best_width:.4f}", 
                color='#333333',
                fontsize=11,
                fontweight='bold',
                ha='left',
                va='center'
            )

            best_patch.set_edgecolor('#333333')
            best_patch.set_linewidth(1.5)

    # Unified Legend
    handles = list(g._legend_data.values())
    labels = list(g._legend_data.keys())

    for rep in ["TF-IDF", "BoW"]:
        style = base_styles[rep]
        line = mlines.Line2D([], [], color=style["color"], linestyle=style["ls"], linewidth=2.5)
        handles.append(line)
        labels.append(f"{rep} Baseline")

    if g._legend:
        g._legend.remove()

    g.fig.legend(handles=handles, labels=labels, loc='center', 
                 bbox_to_anchor=(1, 0.6), title="Representation & Baselines", frameon=True)

    # Styling & Export
    g.set_axis_labels("Macro F1-Score", "Preprocessing Step")
    g.despine(left=True)
    g.fig.suptitle("Ablation Study: Text Preprocessing vs. Model Performance", y=1.02, fontweight="bold")
    g.set(xlim=(0.89, 0.92))

    plt.tight_layout()

    output_dir = Path("results/figures/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_study.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved ablation plot to {output_path}")

    # Cleanup
    plt.close(g.fig)
    del records, results_df, g, baselines, handles, labels
    gc.collect()


if __name__ == "__main__":
    plot_ablation_study()