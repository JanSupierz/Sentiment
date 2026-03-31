import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from src.utils.paths import RESULTS_DIR, FIGURES_DIR

CORRECT_NAVY = '#000080'
ERROR_GREEN = '#00b894'
ERROR_RED = '#d63031'

# 1. Update the palette to use the merged category
CUSTOM_PALETTE = {
    'Correctly Classified': CORRECT_NAVY,
    'False Negative': ERROR_GREEN,
    'False Positive': ERROR_RED,
    'Unknown': '#95a5a6'
}

def process_and_plot_axis(ax, model_filename, title, experiment_name):
    file_path = RESULTS_DIR / "val" / experiment_name / "raw_predictions" / model_filename

    if not file_path.exists():
        ax.text(0.5, 0.5, f"File not found:\n{file_path.name}", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(title, fontsize=18, fontweight='bold')
        return False

    df = pd.read_csv(file_path)
    
    conditions = [
        (df['probability'] <= 0.5) & (df['true_label'] == 0), # True Negative
        (df['probability'] > 0.5) & (df['true_label'] == 1),  # True Positive
        (df['probability'] <= 0.5) & (df['true_label'] == 1), # False Negative
        (df['probability'] > 0.5) & (df['true_label'] == 0)   # False Positive
    ]

    choices = [
        'Correctly Classified',
        'Correctly Classified',
        'False Negative',
        'False Positive'
    ]
    df['Outcome'] = np.select(conditions, choices, default='Unknown')

    hue_order = [
        'Correctly Classified', 
        'False Negative', 
        'False Positive'
    ]

    sns.histplot(
        data=df, x='probability', hue='Outcome', hue_order=hue_order,
        bins=50, multiple="stack", palette=CUSTOM_PALETTE,
        edgecolor='white', linewidth=0.5, kde=False, ax=ax, zorder=2
    )

    fn_rgb = mcolors.to_rgb(ERROR_GREEN)
    fp_rgb = mcolors.to_rgb(ERROR_RED)

    for patch in ax.patches:
        if patch.get_width() == 0:
            continue

        facecolor = patch.get_facecolor()
        
        is_fn = np.allclose(facecolor[:3], fn_rgb, atol=0.05)
        is_fp = np.allclose(facecolor[:3], fp_rgb, atol=0.05)

        if is_fn or is_fp:
            patch.set_hatch('////')
            patch.set_edgecolor((1.0, 1.0, 1.0, 0.5))
            patch.set_linewidth(0.5)

    ax.axvline(0.5, color='#333333', linestyle='--', linewidth=2.5, alpha=0.8, zorder=4)
    ax.text(0.25, ax.get_ylim()[1]*0.95, 'Predicts NEGATIVE', ha='center', va='top', fontsize=14, color='#555555', fontweight='bold', alpha=0.7)
    ax.text(0.75, ax.get_ylim()[1]*0.95, 'Predicts POSITIVE', ha='center', va='top', fontsize=14, color='#555555', fontweight='bold', alpha=0.7)

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Predicted Probability (Positive Class)", fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis='x', visible=False)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    return True

def plot_model_comparison(svm_file, logreg_file, experiment_name):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    process_and_plot_axis(axes[0], svm_file, "Linear SVM (1-3 N-Grams, Z=2.0)", experiment_name)
    process_and_plot_axis(axes[1], logreg_file, "Logistic Regression (1-3 N-Grams, Z=2.0)", experiment_name)

    axes[0].set_ylabel("Number of Samples", fontsize=14, fontweight='bold')
    if len(axes) > 1:
        axes[1].set_ylabel("")

    # 4. Remove the duplicate patch and update ncol to 3
    legend_elements = [
        Patch(facecolor=CORRECT_NAVY, label='Correctly Classified'),
        Patch(facecolor=ERROR_GREEN, hatch='////', edgecolor=(1.0, 1.0, 1.0, 0.5), label='False Negative'),
        Patch(facecolor=ERROR_RED, hatch='////', edgecolor=(1.0, 1.0, 1.0, 0.5), label='False Positive')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

    sns.despine(left=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    out_dir = FIGURES_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison_certainty_dist.png"

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="run_n_grams")
    parser.add_argument("--svm_file", default="ng1-3_linear_svm.csv")
    parser.add_argument("--logreg_file", default="ng1-3_logreg.csv")
    args = parser.parse_args()
    
    plot_model_comparison(args.svm_file, args.logreg_file, args.experiment)