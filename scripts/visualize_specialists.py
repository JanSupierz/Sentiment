import argparse
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from src.utils.paths import PROJECT_ROOT, RESULTS_DIR, FIGURES_DIR

# 1. Standardized Palette from your nice visualization
CORRECT_NAVY = '#000080'
ERROR_GREEN = '#00b894'
ERROR_RED = '#d63031'

CUSTOM_PALETTE = {
    'Correctly Classified': CORRECT_NAVY,
    'False Negative': ERROR_GREEN,
    'False Positive': ERROR_RED,
    'Unknown': '#95a5a6'
}


def plot_axis_from_df(ax, df, title):
    """Handles plotting for a single subplot axis directly from a DataFrame."""
    df = df.copy()
    
    # 2. Map Outcomes using the unified 3-category logic
    conditions = [
        (df['probability'] <= 0.5) & (df['true_label'] == 0), # True Negative
        (df['probability'] > 0.5)  & (df['true_label'] == 1), # True Positive
        (df['probability'] <= 0.5) & (df['true_label'] == 1), # False Negative
        (df['probability'] > 0.5)  & (df['true_label'] == 0)  # False Positive
    ]

    choices = [
        'Correctly Classified',
        'Correctly Classified',
        'False Negative',
        'False Positive'
    ]
    df['Outcome'] = np.select(conditions, choices, default='Unknown')

    # Calculate local accuracy on this specific subset
    correct_mask = df['Outcome'] == 'Correctly Classified'
    accuracy = correct_mask.mean()
    full_title = f"{title}\nAccuracy on subset: {accuracy:.1%}"

    # Draw the Stacked Histogram
    hue_order = [
        'Correctly Classified', 
        'False Negative', 
        'False Positive'
    ]

    sns.histplot(
        data=df, x='probability', hue='Outcome', hue_order=hue_order,
        bins=40, multiple="stack", palette=CUSTOM_PALETTE,
        edgecolor='white', linewidth=0.5, kde=False, ax=ax, zorder=2
    )

    # 3. Apply Hatching to Misclassified (Removed gradient logic)
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

    # Add Decision Boundary
    ax.axvline(0.5, color='#333333', linestyle='--', linewidth=2.5, alpha=0.8, zorder=4)
    
    # Add Predicts Labels
    ax.text(0.25, ax.get_ylim()[1]*0.95, 'Predicts NEGATIVE', ha='center', va='top', fontsize=12, color='#555555', fontweight='bold', alpha=0.7)
    ax.text(0.75, ax.get_ylim()[1]*0.95, 'Predicts POSITIVE', ha='center', va='top', fontsize=12, color='#555555', fontweight='bold', alpha=0.7)

    # Axis Formatting
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis='x', visible=False)
    ax.set_xlim(0, 1.0)

    if ax.get_legend() is not None:
        ax.get_legend().remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    # Load config for threshold and base model type
    with open(PROJECT_ROOT / args.best_params) as f:
        cfg = yaml.safe_load(f)

    threshold = cfg.get('cascade', {}).get('delegation_threshold', 0.15)
    base_model_type = cfg.get('model', 'linear_svm')
    base_name = "Base_SVM" if base_model_type == 'linear_svm' else "Base_LogReg"

    # Define paths
    pred_dir = RESULTS_DIR / "test" / "raw_predictions"
    paths = {
        "Base": pred_dir / f"{base_name}.csv",
        "SVM_Spec": pred_dir / "SVM_Specialist.csv",
        "BERT_Basic": pred_dir / "BERT_Basic.csv",
        "BERT_Spec": pred_dir / "BERT_Specialist.csv"
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found. Ensure you have run the inference script.")

    # Load all dataframes
    df_base = pd.read_csv(paths["Base"])
    df_svm_spec = pd.read_csv(paths["SVM_Spec"])
    df_bert_basic = pd.read_csv(paths["BERT_Basic"])
    df_bert_spec = pd.read_csv(paths["BERT_Spec"])

    lower = threshold
    upper = 1.0 - threshold

    # Find the indices where the BASE model was uncertain
    mask_hard = (df_base['probability'] >= lower) & (df_base['probability'] <= upper)

    hard_base = df_base[mask_hard]
    hard_svm_spec = df_svm_spec[mask_hard]
    hard_bert_basic = df_bert_basic[mask_hard]
    hard_bert_spec = df_bert_spec[mask_hard]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 4, figsize=(32, 7), sharey=True)

    fig.suptitle(f"Model Behaviors on the Hardest {mask_hard.mean():.1%} of Data\n"
                 f"(Samples where Base Probability was between {lower:.2f} and {upper:.2f})", 
                 fontsize=20, fontweight='bold', y=1.08)

    plot_axis_from_df(axes[0], hard_base, "1. Original Base Model")
    plot_axis_from_df(axes[1], hard_svm_spec, "2. SVM Specialist")
    plot_axis_from_df(axes[2], hard_bert_basic, "3. Basic BERT")
    plot_axis_from_df(axes[3], hard_bert_spec, "4. BERT Specialist")

    axes[0].set_ylabel("Number of Hard Samples", fontsize=14, fontweight='bold')
    for ax in axes[1:]:
        ax.set_ylabel("")

    # 4. Create a unified 3-column legend for the entire figure
    legend_elements = [
        Patch(facecolor=CORRECT_NAVY, label='Correctly Classified'),
        Patch(facecolor=ERROR_GREEN, hatch='////', edgecolor=(1.0, 1.0, 1.0, 0.5), label='False Negative'),
        Patch(facecolor=ERROR_RED, hatch='////', edgecolor=(1.0, 1.0, 1.0, 0.5), label='False Positive')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

    sns.despine(left=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save the figure
    out_dir = FIGURES_DIR / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_models_hard_cases_analysis.png"

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved analysis plot to: {out_path}")


if __name__ == "__main__":
    main()