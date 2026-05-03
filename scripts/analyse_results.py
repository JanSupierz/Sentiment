import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import yaml
from dataclasses import dataclass

from src.utils.paths import RESULTS_DIR, FIGURES_DIR, CONFIGS_DIR

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


@dataclass(frozen=True)
class PlotConfig:
    """Struct to hold plot styling parameters."""
    title_size: int = 22
    subtitle_size: int = 18
    label_size: int = 15
    tick_size: int = 15
    legend_size: int = 15


def sweep_exact_cascade(svm_df, bert_preds, y_true):
    certainty_thresholds = np.linspace(0.500, 1.0, 200)
    svm_probs = svm_df['probability'].values
    svm_preds = (svm_probs > 0.5).astype(int)

    results = []
    for c in certainty_thresholds:
        lower_bound = 1.0 - c

        if c >= 1.0:
            certain_mask = np.zeros_like(svm_probs, dtype=bool)
        else:
            certain_mask = (svm_probs >= c) | (svm_probs <= lower_bound)

        delegated_mask = ~certain_mask
        cov = certain_mask.mean()
        delegated = 1.0 - cov

        # LOCAL ACCURACY (What the base model scores on its retained portion)
        if cov > 0:
            local_acc = (svm_preds[certain_mask] == y_true[certain_mask]).mean()
        else:
            local_acc = 1.0

        # GLOBAL HYBRID ACCURACY
        svm_correct = (svm_preds[certain_mask] == y_true[certain_mask]).sum()
        bert_correct = (bert_preds[delegated_mask] == y_true[delegated_mask]).sum()
        hybrid_acc = (svm_correct + bert_correct) / len(y_true)
        results.append((c, cov, delegated, local_acc, hybrid_acc))

    return pd.DataFrame(results, columns=['Certainty_Threshold', 'Coverage', 'Delegated', 'Local_Accuracy', 'Hybrid_Accuracy'])


def plot_cascade_metrics(df_subset, model_name_label, bert_global_acc, fig_dir, style: PlotConfig):

    if df_subset.empty:
        print(f"Skipping plots for {model_name_label}: No data found.")
        return

    # Find the peak performance for the star marker within this subset
    best_row = df_subset.loc[df_subset['Hybrid_Accuracy'].idxmax()]
    best_delegated = best_row['Delegated']
    best_hybrid_acc = best_row['Hybrid_Accuracy']

    # Filter down to the top 4 models
    top_models = df_subset.groupby('Model')['Hybrid_Accuracy'].max().sort_values(ascending=False).head(4).index.tolist()
    plot_df = df_subset[df_subset['Model'].isin(top_models)].copy()

    # Clean up model names for the legend
    plot_df['Model'] = plot_df['Model'].apply(lambda x: x.replace("linear_svm", "SVM").replace("logreg", "LR"))
    clean_top_models = [m.replace("linear_svm", "SVM").replace("logreg", "LR") for m in top_models]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    colors = sns.color_palette("tab10", len(clean_top_models))
    palette = {m: c for m, c in zip(clean_top_models, colors)}

    # --- Graph 1: Local Accuracy vs Certainty Requirement ---
    sns.lineplot(data=plot_df, x="Certainty_Threshold", y="Local_Accuracy", hue="Model", 
                 style="Model", dashes=True, alpha=0.8,
                 palette=palette, ax=ax1, linewidth=2.5, legend=False)
    ax1.set_title("1. Local Accuracy on Retained Data", fontsize=style.subtitle_size)
    ax1.set_xlabel(r"Model Prediction Certainty (%)", fontsize=style.label_size)
    ax1.set_ylabel("Local Accuracy (%)", fontsize=style.label_size)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='both', which='major', labelsize=style.tick_size)
    ax1.set_xlim(0.5, 1.0) 
    ymin, _ = ax1.get_ylim()
    ax1.set_ylim(ymin, 1.0)

    # --- Graph 2: Deferral Curve (Data Delegated vs Certainty) ---
    sns.lineplot(data=plot_df, x="Certainty_Threshold", y="Delegated", hue="Model", 
                 style="Model", dashes=True, alpha=0.8,
                 palette=palette, ax=ax2, linewidth=2.5, legend=False)
    ax2.set_title("2. Workload Management (Deferral Curve)", fontsize=style.subtitle_size)
    ax2.set_xlabel(r"Model Prediction Certainty (%)", fontsize=style.label_size)
    ax2.set_ylabel("Data Delegated to BERT (%)", fontsize=style.label_size)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.tick_params(axis='both', which='major', labelsize=style.tick_size)
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(0.0, 1.0)

    # --- Graph 3: True Hybrid Optimization Score ---
    sns.lineplot(data=plot_df, x="Delegated", y="Hybrid_Accuracy", hue="Model", 
                 style="Model", dashes=True, alpha=0.8,
                 palette=palette, ax=ax3, linewidth=2.5)

    ax3.axhline(bert_global_acc, color='black', linestyle='--', linewidth=2, label=f"BERT Global Baseline ({bert_global_acc:.1%})")
    ax3.plot(best_delegated, best_hybrid_acc, marker='*', markersize=18, color='gold', markeredgecolor='black', zorder=10, label=f"Optimal {model_name_label} Peak")

    ax3.set_title("3. Global System Optimization Score", fontsize=style.subtitle_size)
    ax3.set_xlabel("Data Delegated to BERT (%)", fontsize=style.label_size)
    ax3.set_ylabel("Combined Cascade Accuracy (%)", fontsize=style.label_size)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.tick_params(axis='both', which='major', labelsize=style.tick_size)

    ax3.legend(fontsize=style.legend_size, loc='upper right', bbox_to_anchor=(1.0, 0.95))
    ax3.set_xlim(0.0, 1.0)

    plt.tight_layout()
    filename = f"{model_name_label.lower()}_accuracy_coverage_tradeoff.png"
    plot_path = fig_dir / filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved {model_name_label} Trade-off graphs to: {plot_path}")
    plt.close() # Close figure to free memory

    try:
        from IPython.display import display, Image
        display(Image(filename=plot_path))
    except ImportError:
        pass


def main(experiments, bert_exp):
    raw_base = RESULTS_DIR / "val"
    fig_dir = FIGURES_DIR / "analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load Ground Truth and BERT Baseline exact predictions
    bert_file = raw_base / bert_exp / "raw_predictions" / "bert_basic_baseline.csv"
    if not bert_file.exists():
        print(f"BERT baseline CSV not found at {bert_file}. Run train_bert.py first.")
        return

    bert_df = pd.read_csv(bert_file)
    y_true = bert_df['true_label'].values
    bert_preds = (bert_df['probability'].values > 0.5).astype(int)
    bert_global_acc = (bert_preds == y_true).mean()

    # Gather CSVs from the specified experiment folders
    csv_files = []
    for exp in experiments:
        exp_dir = raw_base / exp / "raw_predictions"
        if exp_dir.exists():
            csv_files.extend([f for f in exp_dir.glob("*.csv") if f.name != "bert_basic_baseline.csv"])
        else:
            print(f"Warning: Directory {exp_dir} not found. Skipping.")

    if not csv_files:
        print("No model prediction CSVs found in the specified experiments!")
        return

    all_results = []
    print(f"\nSweeping Certainty Thresholds for Exact Hybrid Optimization...")

    for f in csv_files:
        model_name = f.stem
        svm_df = pd.read_csv(f)

        if len(svm_df) != len(y_true):
            print(f"Warning: {f.name} length mismatch. Skipping.")
            continue

        metrics_df = sweep_exact_cascade(svm_df, bert_preds, y_true)
        metrics_df['Model'] = model_name
        all_results.append(metrics_df)

    curve_df = pd.concat(all_results, ignore_index=True)

    # Max Hybrid Accuracy
    best_row = curve_df.loc[curve_df['Hybrid_Accuracy'].idxmax()]

    winner_name = best_row['Model']
    best_certainty = best_row['Certainty_Threshold']
    best_delegated = best_row['Delegated']
    best_hybrid_acc = best_row['Hybrid_Accuracy']
    lower_bound = 1.0 - best_certainty

    parts = winner_name.split('_')

    base_m = "linear_svm" if "svm" in winner_name.lower() else "logreg"
    ng_part = next((p for p in parts if p.startswith('ng')), "ng1-3")
    ng_str = ng_part.replace('ng', '').split('-')
    k_part = next((p for p in parts if p.startswith('k')), "k0")
    k_val = int(k_part.replace('k', ''))
    w_part = next((p for p in parts if p.startswith('w')), "w0")
    w_val = float(w_part.replace('w', ''))
    z_part = next((p for p in parts if p.lower().startswith('z')), "z2.0")
    z_val = float(z_part.lower().replace('z', ''))

    # Determine data token column from filename
    token_col = "tokens_lower"
    known_data_names = ['text_raw', 'text_clean', 'text_expanded',
                        'tokens_cased', "tokens_lower", 'tokens_filtered',
                        'tokens_letters', 'tokens_stemmed', 'tokens_lemmatized']

    for name in known_data_names:
        if name in winner_name:
            token_col = name
            break

    best_params = {
        "model": base_m,
        "features": {
            "token_col": token_col,
            "ngram_range": [int(ng_str[0]), int(ng_str[1])],
            "n_concepts": k_val,
            "z_threshold": z_val,
            "use_concepts": k_val > 0,
            "sentiment_weight": w_val
        },
        "cascade": {"delegation_threshold": round(float(lower_bound), 3)},
    }

    out_yaml = CONFIGS_DIR / "best_params.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(best_params, f)

    print(f"Auto-saved optimal configuration to {out_yaml.name}")

    print(f"\nCASCADE OPTIMIZATION WINNER: {winner_name}")
    print(f"-> Peak Global System Accuracy: {best_hybrid_acc:.2%} (Beats Base BERT by {best_hybrid_acc - bert_global_acc:+.2%})")
    print(f"-> Required Model Certainty: {best_certainty:.1%} (Accepts probs >= {best_certainty:.3f} or <= {lower_bound:.3f})")
    print(f"-> Data Delegated to BERT: {best_delegated:.2%} (Base model handles {(1-best_delegated):.2%})\n")

    # Initialize styling struct
    style = PlotConfig()

    # Filter for SVM models and plot
    svm_df_subset = curve_df[curve_df['Model'].str.contains('svm', case=False, na=False)]
    plot_cascade_metrics(svm_df_subset, "SVM", bert_global_acc, fig_dir, style)

    # Filter for LogReg models and plot
    logreg_df_subset = curve_df[curve_df['Model'].str.contains('logreg', case=False, na=False)]
    plot_cascade_metrics(logreg_df_subset, "LogReg", bert_global_acc, fig_dir, style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep cascade thresholds across multiple experiment runs.")
    parser.add_argument("--experiments", nargs="+", default=["grid_search", "run_n_grams", "run_basic", "run_custom", "run_on_preprocessing"], 
                        help="List of experiment folders to pull SVM/LogReg CSVs from (e.g. grid_search run_n_grams)")
    parser.add_argument("--bert_exp", default="bert",
                        help="The experiment folder containing your bert_basic_baseline.csv")

    args = parser.parse_args()
    main(args.experiments, args.bert_exp)