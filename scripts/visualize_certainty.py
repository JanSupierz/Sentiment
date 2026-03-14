import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from src.utils.paths import RESULTS_DIR, FIGURES_DIR

correct_cmap = mcolors.LinearSegmentedColormap.from_list("red_green", ["#e74c3c", "#f1c40f", "#2ecc71"])
tn_base_color = mcolors.to_hex(correct_cmap(0.25))
tp_base_color = mcolors.to_hex(correct_cmap(0.75))
error_green = '#00b894'
error_red = '#d63031'

custom_palette = {
    'True Negative (Correct)': tn_base_color,
    'True Positive (Correct)': tp_base_color,
    'False Negative (Incorrect)': error_green,
    'False Positive (Incorrect)': error_red,
    'Unknown': '#95a5a6'
}


def process_and_plot_axis(ax, model_filename, title, experiment_name):
    """Handles the data loading and plotting for a single subplot axis."""
    raw_dir = RESULTS_DIR / "val" / experiment_name / "raw_predictions"
    file_path = raw_dir / model_filename

    if not file_path.exists():
        ax.text(0.5, 0.5, f"File not found:\n{file_path.name}", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(title, fontsize=18, fontweight='bold')
        return False

    # Load the Prediction Data
    df = pd.read_csv(file_path)
    df['Outcome'] = 'Unknown'
    df.loc[(df['probability'] <= 0.5) & (df['true_label'] == 0), 'Outcome'] = 'True Negative (Correct)'
    df.loc[(df['probability'] > 0.5)  & (df['true_label'] == 1), 'Outcome'] = 'True Positive (Correct)'
    df.loc[(df['probability'] <= 0.5) & (df['true_label'] == 1), 'Outcome'] = 'False Negative (Incorrect)'
    df.loc[(df['probability'] > 0.5)  & (df['true_label'] == 0), 'Outcome'] = 'False Positive (Incorrect)'

    # Draw the Stacked Histogram
    hue_order = ['True Negative (Correct)', 'True Positive (Correct)', 'False Negative (Incorrect)', 'False Positive (Incorrect)']

    sns.histplot(
        data=df, x='probability', hue='Outcome', hue_order=hue_order,
        bins=50, multiple="stack", palette=custom_palette,
        edgecolor='white', linewidth=0.5, kde=False, ax=ax, zorder=2
    )

    # Apply Gradient to Correct Predictions & Hatching to Misclassified
    tn_rgb = mcolors.to_rgb(tn_base_color)
    tp_rgb = mcolors.to_rgb(tp_base_color)
    fn_rgb = mcolors.to_rgb(error_green)
    fp_rgb = mcolors.to_rgb(error_red)

    for patch in ax.patches:
        if patch.get_width() == 0:
            continue

        facecolor = patch.get_facecolor()
        is_tn = all(abs(facecolor[i] - tn_rgb[i]) < 0.05 for i in range(3))
        is_tp = all(abs(facecolor[i] - tp_rgb[i]) < 0.05 for i in range(3))
        is_fn = all(abs(facecolor[i] - fn_rgb[i]) < 0.05 for i in range(3))
        is_fp = all(abs(facecolor[i] - fp_rgb[i]) < 0.05 for i in range(3))

        if is_tn or is_tp:
            x_center = patch.get_x() + patch.get_width() / 2.0
            new_rgb = correct_cmap(np.clip(x_center, 0, 1))
            patch.set_facecolor((*new_rgb[:3], facecolor[3]))

        elif is_fn or is_fp:
            patch.set_hatch('////')
            patch.set_edgecolor((1.0, 1.0, 1.0, 0.5))
            patch.set_linewidth(0.5)

    # Add Decision Boundary & Annotations
    ax.axvline(0.5, color='#333333', linestyle='--', linewidth=2.5, alpha=0.8, zorder=4)
    ax.text(0.25, ax.get_ylim()[1]*0.95, 'Predicts NEGATIVE', ha='center', va='top', fontsize=14, color='#555555', fontweight='bold', alpha=0.7)
    ax.text(0.75, ax.get_ylim()[1]*0.95, 'Predicts POSITIVE', ha='center', va='top', fontsize=14, color='#555555', fontweight='bold', alpha=0.7)

    # Axis Formatting
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Predicted Probability (Positive Class)", fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis='x', visible=False)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    return True


def plot_model_comparison(svm_file, logreg_file, experiment_name):
    sns.set_theme(style="whitegrid", context="talk")

    # Create a 1x2 grid of subplots sharing the Y-axis
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    process_and_plot_axis(axes[0], svm_file, "Linear SVM (1-3 N-Grams, Z=2.0)", experiment_name)
    process_and_plot_axis(axes[1], logreg_file, "Logistic Regression (1-3 N-Grams, Z=2.0)", experiment_name)

    # Only the leftmost plot needs the Y-axis label
    axes[0].set_ylabel("Number of Samples", fontsize=14, fontweight='bold')
    if len(axes) > 1:
        axes[1].set_ylabel("")

    # Create a unified legend for the entire figure
    legend_elements = [
        Patch(facecolor=tn_base_color, label='True Negative (Correct)'),
        Patch(facecolor=error_green, hatch='////', edgecolor=(1.0, 1.0, 1.0, 0.5), label='False Negative (Incorrect)'),
        Patch(facecolor=tp_base_color, label='True Positive (Correct)'),
        Patch(facecolor=error_red, hatch='////', edgecolor=(1.0, 1.0, 1.0, 0.5), label='False Positive (Incorrect)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)

    sns.despine(left=True)
    plt.tight_layout()
    # Adjust layout to make room for the legend at the bottom
    plt.subplots_adjust(bottom=0.15)

    # Save the figure
    out_dir = FIGURES_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison_certainty_dist.png"

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved comparison plot to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="run_n_grams", help="Name of the experiment folder (e.g., run_n_grams)")
    parser.add_argument("--svm_file", default="ng1-3_linear_svm.csv")
    parser.add_argument("--logreg_file", default="ng1-3_logreg.csv")
    args = parser.parse_args()
    
    plot_model_comparison(args.svm_file, args.logreg_file, args.experiment)