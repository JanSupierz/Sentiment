import argparse
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass

from src.utils.paths import PROJECT_ROOT, RESULTS_DIR, FIGURES_DIR


@dataclass(frozen=True)
class PlotConfig:
    """Struct to hold plot styling parameters."""
    font_size: int = 15
    title_size: int = 18  # font_size + 3 for hierarchy
    pad: int = 15
    cmap: str = 'Blues'


def plot_cm_from_df(ax, df, title, style: PlotConfig):
    """Handles plotting a confusion matrix for a single subplot axis directly from a DataFrame."""
    df = df.copy()
    
    # Binarize predictions based on the 0.5 threshold
    y_true = df['true_label']
    y_pred = (df['probability'] > 0.5).astype(int)
    
    # Calculate accuracy for the title
    accuracy = (y_true == y_pred).mean()
    full_title = f"{title}\nAccuracy on subset: {accuracy:.1%}"

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Draw the Heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=style.cmap, ax=ax, 
        cbar=False, square=True,
        xticklabels=['Negative (0)', 'Positive (1)'],
        yticklabels=['Negative (0)', 'Positive (1)'],
        annot_kws={"size": style.font_size, "weight": "bold"}
    )

    # Axis Formatting
    ax.set_title(full_title, fontsize=style.font_size, fontweight='bold', pad=style.pad)
    ax.set_xlabel("Predicted Label", fontsize=style.font_size, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=style.font_size, fontweight='bold')


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

    # Initialize styling struct
    style = PlotConfig()

    sns.set_theme(style="white", context="talk")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    plot_cm_from_df(axes[0], hard_base, "1. Original Base Model", style)
    plot_cm_from_df(axes[1], hard_svm_spec, "2. SVM Specialist", style)
    plot_cm_from_df(axes[2], hard_bert_basic, "3. Basic BERT", style)
    plot_cm_from_df(axes[3], hard_bert_spec, "4. BERT Specialist", style)

    # Only show the y-axis label on the first plot to avoid clutter
    for ax in axes[1:]:
        ax.set_ylabel("")

    plt.tight_layout()

    # Save the figure
    out_dir = FIGURES_DIR / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_models_hard_cases_cm.png"

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved confusion matrix plot to: {out_path}")


if __name__ == "__main__":
    main()