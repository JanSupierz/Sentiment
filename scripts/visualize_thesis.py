import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(".")
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

def plot_cascade_specialists_only():
    input_csv = RESULTS_DIR / "thesis" / "cascade_simulation_results.csv"
    out_plot = RESULTS_DIR / FIGURES_DIR / "thesis" / "cascade_comparison_f1_macro.png"

    if not input_csv.exists():
        print(f"Error: {input_csv} not found.")
        return

    NAME_MAP = {
        "BERT_Basic_Cascade": "Base + Basic BERT",
        "BERT_Spec_Cascade": "Base + Spec. BERT",
        "SVM_Spec_Cascade": "Base + Spec. SVM",
        "BERT_Basic": "BERT Basic"
    }

    df = pd.read_csv(input_csv)
    row = df.iloc[0]
    threshold = row['Threshold']
    delegation_rate = row['Delegation_Rate']

    predictions_dir = RESULTS_DIR / "test" / "raw_predictions"

    baseline_file = predictions_dir / "Base_SVM.csv"
    if not baseline_file.exists():
        print(f"Error: Baseline file {baseline_file} not found.")
        return
    baseline_df = pd.read_csv(baseline_file)
    y_true = baseline_df['true_label'].values
    preds_base = (baseline_df['probability'] > 0.5).astype(int)
    baseline_macro_f1 = f1_score(y_true, preds_base, average='macro')

    specialist_files = [
        "BERT_Basic_Cascade",
        "BERT_Spec_Cascade",
        "SVM_Spec_Cascade",
        "BERT_Basic"
    ]

    specialist_scores = []
    significant_improvements = set()

    for spec_name in specialist_files:
        spec_file = predictions_dir / f"{spec_name}.csv"
        if not spec_file.exists():
            print(f"Warning: specialist file {spec_file} not found. Skipping.")
            continue

        spec_df = pd.read_csv(spec_file)
        preds_spec = (spec_df['probability'] > 0.5).astype(int)
        macro_f1 = f1_score(y_true, preds_spec, average='macro')
        specialist_scores.append((spec_name, macro_f1))

        both_correct = np.sum((preds_base == y_true) & (preds_spec == y_true))
        base_only     = np.sum((preds_base == y_true) & (preds_spec != y_true))
        spec_only     = np.sum((preds_base != y_true) & (preds_spec == y_true))
        both_wrong    = np.sum((preds_base != y_true) & (preds_spec != y_true))
        table = [[both_correct, base_only],
                 [spec_only,    both_wrong]]
        p_val = mcnemar(table, exact=False, correction=True).pvalue

        if p_val < 0.05:
            mapped_label = NAME_MAP.get(spec_name, spec_name)
            significant_improvements.add(mapped_label)
            print(f"Significant: {mapped_label} (p={p_val:.4f})")

    if not specialist_scores:
        print("No specialist scores computed. Exiting.")
        return

    plot_labels = [fn for fn, _ in specialist_scores]
    f1_values = [val for _, val in specialist_scores]

    sns.set_theme(style="white", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    BLUE = "#3498db"
    GREY = "#bdc3c7"
    NAVY = "#2c3e50"

    colors = [BLUE if NAME_MAP.get(lbl, lbl) in significant_improvements else GREY for lbl in plot_labels]

    bars = ax.bar(plot_labels, f1_values, color=colors,
                  edgecolor='white', linewidth=1.5, zorder=3, width=0.6)

    ax.axhline(y=baseline_macro_f1, color=NAVY, linestyle='--', linewidth=2.5, zorder=4, alpha=0.8)
    ax.text(0.01, baseline_macro_f1 + 0.0005,
            f'BASELINE (Macro F1: {baseline_macro_f1:.4f})',
            transform=ax.get_yaxis_transform(),
            color=NAVY, fontweight='bold', va='bottom', fontsize=12)

    max_f1 = max(f1_values)
    for bar, lbl, f1_val in zip(bars, plot_labels, f1_values):
        is_sig = NAME_MAP.get(lbl, lbl) in significant_improvements

        prefix = "★ " if f1_val == max_f1 else ""
        t_color = BLUE if is_sig else "#7f8c8d"
        f_weight = 'bold' if is_sig else 'normal'

        ax.annotate(f'{prefix}{f1_val:.4f}',
                    (bar.get_x() + bar.get_width() / 2., f1_val),
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', fontsize=11,
                    color=t_color, fontweight=f_weight)

    ax.set_ylabel("Macro F1 Score", fontsize=14, fontweight='bold')

    display_labels = [NAME_MAP.get(lbl, lbl) for lbl in plot_labels]
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=0, fontsize=12)

    y_min = max(0, min(baseline_macro_f1, *f1_values) - 0.01)
    y_max = min(1.0, max(max_f1, baseline_macro_f1) + 0.015)
    ax.set_ylim(y_min, y_max)

    sns.despine(left=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    legend_elements = [
        plt.Line2D([0], [0], color=NAVY, linestyle='--', lw=2, label='SVM Baseline'),
        mpatches.Patch(color=BLUE, label='Statistically Significant (p < 0.05)'),
        mpatches.Patch(color=GREY, label='Insignificant')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', shadow=True)

    plt.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Wykres zapisany w: {out_plot}")

if __name__ == "__main__":
    plot_cascade_specialists_only()