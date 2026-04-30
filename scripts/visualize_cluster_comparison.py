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
                "k": k,
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
    results_df = results_df.sort_values(by="k")

    # --- AESTHETIC UPGRADES START HERE ---
    
    # 1. Set a clean theme with horizontal grid lines
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.15", "axes.linewidth": 1.25})
    plt.rcParams.update({'font.size': 14})

    weights_order = sorted(results_df["Sentiment Weight (w)"].unique(),
                           key=lambda x: float(x.split('=')[1]))
    palette = sns.color_palette("viridis", len(weights_order))

    # 2. Enhanced dot aesthetics (larger size, white borders for contrast)
    g = sns.relplot(
        data=results_df,
        x="k",
        y="Macro F1-Score",
        hue="Sentiment Weight (w)",
        hue_order=weights_order,
        palette=palette,
        col="Model",
        kind="scatter",
        height=7,        # Slightly reduced height for better proportions
        aspect=1.25,
        legend=False,
        s=130,           # Increased marker size
        edgecolor="white", # Added white border to dots
        linewidth=1.2,     # Border thickness
        alpha=0.9
    )

    g.set_titles("{col_name}", size=16, pad=15)
    g.set_axis_labels("Number of Concepts (k)", "Macro F1-Score")

    # 3. Force X-axis to only show ticks where you actually have data
    unique_k_values = sorted(results_df['k'].unique())
    g.set(xticks=unique_k_values)

    # Add baseline horizontal lines with slight transparency
    for model_label, ax in g.axes_dict.items():
        if model_label in baselines:
            ax.axhline(y=baselines[model_label], color="coral",
                       linestyle="--", linewidth=2.5, alpha=0.8)

    # Custom legend
    legend_handles = []
    for w, col in zip(weights_order, palette):
        handle = mlines.Line2D([], [], marker='o', color='white',
                               markerfacecolor=col, markersize=11,
                               linestyle='None', label=w)
        legend_handles.append(handle)

    baseline_handle = mlines.Line2D([], [], color="coral", linestyle="--",
                                    linewidth=2.5, alpha=0.8,
                                    label=f"Z={z_fixed} Baseline")
    legend_handles.append(baseline_handle)

    g.fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.98, 0.5), # Tucked slightly closer to the plot
        title="Sentiment Weight",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8"
    )

    # 4. Proportional X-axis limits instead of +2
    max_k = results_df['k'].max()
    padding = max_k * 0.05  # 5% padding
    g.set(xlim=(-padding, max_k + padding))

    # Remove top/right spines, keep bottom/left
    g.despine(left=False, bottom=False) 

    # ----- Export as PDF with dpi=300 -----
    output_dir = Path("results/figures/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cluster_comparison.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved aesthetically improved plot to {out_path}")

if __name__ == "__main__":
    plot_cluster_comparison_polished()