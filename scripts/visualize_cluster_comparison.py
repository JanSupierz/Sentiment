import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path


def plot_cluster_comparison_polished(reports_dir="results/val/grid_search/classification_reports"):
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        print(f"Could not find directory: {reports_path}")
        return

    z_fixed = 2.0

    # Load Baselines
    baselines = {}
    for model_key, label in [("linear_svm", "Linear SVM"), ("logreg", "Logistic Regression")]:

        base_file = reports_path / f"{model_key}_k0_w0.csv"
        if base_file.exists():
            df = pd.read_csv(base_file, index_col=0)
            baselines[label] = df.loc["macro avg", "f1-score"]
        else:
            print(f"Warning: Could not find baseline file {base_file.name}")

    # Load Grid Search Cluster Results
    records = []
    for file in reports_path.glob("*.csv"):
        stem = file.stem
        if "Custom" in stem or "baseline" in stem:
            continue

        parts = stem.split('_')
        k_part = next((p for p in parts if p.startswith('k')), None)
        w_part = next((p for p in parts if p.startswith('w')), None)

        if not (k_part and w_part):
            continue

        try:
            k = int(k_part[1:])
            w = float(w_part[1:])

            if k == 0:
                continue

            idx_k = parts.index(k_part)
            model_str = '_'.join(parts[:idx_k])
            model_label = "Linear SVM" if "svm" in model_str else "Logistic Regression"

            df = pd.read_csv(file, index_col=0)
            records.append({
                "Number of Concepts (k)": f"k={k}",
                "Sentiment Weight (w)": f"w={int(w) if w.is_integer() else w}",
                "Model": model_label,
                "Macro F1-Score": df.loc["macro avg", "f1-score"]
            })
        except Exception:
            continue

    if not records:
        print("No cluster configuration files found to plot!")
        return

    results_df = pd.DataFrame(records)

    # Sort for a clean Y-axis progression
    results_df["k_sort"] = results_df["Number of Concepts (k)"].apply(lambda x: int(x.replace('k=', '')))
    results_df = results_df.sort_values(by="k_sort").drop(columns=["k_sort"])

    # Create the Ablation-style Plot
    sns.set_theme(style="whitegrid", context="talk")

    g = sns.catplot(
        data=results_df,
        kind="bar",
        x="Macro F1-Score",
        y="Number of Concepts (k)",
        hue="Sentiment Weight (w)",
        col="Model",
        height=8,
        aspect=1.2,
        palette="viridis",
        alpha=0.9
    )

    g.set_titles("{col_name}")

    # Add Baseline Markers & Legend
    handles = list(g._legend_data.values()) if g._legend_data else []
    labels = list(g._legend_data.keys()) if g._legend_data else []

    baseline_added = False
    for ax in g.axes.flat:
        title = ax.get_title()
        if title in baselines:
            score = baselines[title]
            ax.axvline(x=score, color="coral", linestyle="--", linewidth=2.5)
            if not baseline_added:
                line = mlines.Line2D([], [], color="coral", linestyle="--", linewidth=2.5)
                handles.append(line)
                labels.append(f"Z={z_fixed} Baseline (No Clusters)")
                baseline_added = True

    if g._legend:
        g._legend.remove()

    g.fig.legend(
        handles=handles, labels=labels,
        loc='center', bbox_to_anchor=(1, 0.6),
        title="Weights & Baselines", frameon=True
    )

    # Styling & Export
    g.set_axis_labels("Macro F1-Score", "Number of Concepts (k)")
    g.despine(left=True)
    g.fig.suptitle("Grid Search: Clustered Concepts vs. Standard Processing", y=1.02, fontweight="bold")

    min_val = min(results_df["Macro F1-Score"].min(), min(baselines.values()) if baselines else 1)
    max_val = max(results_df["Macro F1-Score"].max(), max(baselines.values()) if baselines else 0)
    g.set(xlim=(min_val - 0.005, max_val + 0.005))

    plt.tight_layout()

    output_dir = Path("results/figures/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cluster_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved refined cluster plot to {out_path}")


if __name__ == "__main__":
    plot_cluster_comparison_polished()