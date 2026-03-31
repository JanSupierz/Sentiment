import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(".")
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def plot_cascade_specialists_only():
    input_csv = RESULTS_DIR / "thesis" / "cascade_simulation_results.csv"
    mcnemar_csv_path = RESULTS_DIR / "analysis" / "cascade_mcnemar_raport.csv"
    out_plot = RESULTS_DIR / FIGURES_DIR / "thesis" / "cascade_comparison_f1.png"

    if not input_csv.exists():
        print(f"Error: {input_csv} not found.")
        return

    NAME_MAP = {
        "BERT_Basic_Cascade": "2. Base + Basic BERT",
        "BERT_Spec_Cascade": "3. Base + Spec. BERT",
        "SVM_Spec_Cascade": "4. Base + Spec. SVM"
    }

    # 3. Loading F1 Data
    df = pd.read_csv(input_csv)
    row = df.iloc[0]

    threshold = row['Threshold']
    delegation_rate = row['Delegation_Rate']

    metadata_cols = ['Threshold', 'Delegation_Rate']
    all_data_cols = [c for c in df.columns if c not in metadata_cols]

    baseline_f1 = row[all_data_cols[0]]
    specialist_cols = all_data_cols[1:]

    plot_data = row[specialist_cols].to_frame().reset_index()
    plot_data.columns = ['Model', 'F1_Score']

    significant_improvements = set()
    if mcnemar_csv_path.exists():
        df_mc = pd.read_csv(mcnemar_csv_path)

        for _, mc_row in df_mc.iterrows():
            p_val = mc_row.get('P_Value', 1.0)
            csv_model_name = mc_row.get('Compared_Model', '')

            if p_val < 0.05:
                mapped_label = NAME_MAP.get(csv_model_name)
                if mapped_label:
                    significant_improvements.add(mapped_label)
                    print(f"Significant: {mapped_label} (p={p_val:.4f})")
    else:
        print(f"Warning: {mcnemar_csv_path} not found.")

    sns.set_theme(style="white", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#F8F9FA')
    ax.set_facecolor('white')

    BLUE = "#3498db" # Significant
    GREY = "#bdc3c7" # Not significant
    NAVY = "#2c3e50" # Baseline

    colors = [BLUE if m in significant_improvements else GREY for m in plot_data['Model']]

    bars = ax.bar(plot_data['Model'], plot_data['F1_Score'], color=colors, 
                  edgecolor='white', linewidth=1.5, zorder=3, width=0.6)

    ax.axhline(y=baseline_f1, color=NAVY, linestyle='--', linewidth=2.5, zorder=4, alpha=0.8)
    ax.text(0.01, baseline_f1 + 0.0005, f' BASELINE (F1: {baseline_f1:.4f})', 
            transform=ax.get_yaxis_transform(),
            color=NAVY, fontweight='bold', va='bottom', fontsize=12)

    max_f1 = plot_data['F1_Score'].max()
    for bar, model_name in zip(bars, plot_data['Model']):
        height = bar.get_height()
        is_sig = model_name in significant_improvements

        prefix = "★ " if height == max_f1 else ""
        t_color = BLUE if is_sig else "#7f8c8d"
        f_weight = 'bold' if is_sig else 'normal'

        ax.annotate(f'{prefix}{height:.4f}',
                    (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', fontsize=11,
                    color=t_color, fontweight=f_weight)

    ax.set_title(f"Performance Comparison: Baseline vs. Cascade Specialists\n"
                 f"Threshold: {threshold:.2f} | Delegation Rate: {delegation_rate:.1%}", 
                 fontsize=18, fontweight='bold', pad=30)

    ax.set_ylabel("Weighted F1 Score", fontsize=14, fontweight='bold')

    display_labels = [l.split('. ', 1)[-1] for l in plot_data['Model']]
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=0, fontsize=12)

    y_min = max(0, min(baseline_f1, plot_data['F1_Score'].min()) - 0.01)
    y_max = min(1.0, max(max_f1, baseline_f1) + 0.015)
    ax.set_ylim(y_min, y_max)

    sns.despine(left=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    legend_elements = [
        plt.Line2D([0], [0], color=NAVY, linestyle='--', lw=2, label='Original Base Model'),
        mpatches.Patch(color=BLUE, label='Statistically Significant (p < 0.05)'),
        mpatches.Patch(color=GREY, label='Insignificant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', shadow=True)

    plt.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    print(f"Wykres zapisany w: {out_plot}")


if __name__ == "__main__":
    plot_cascade_specialists_only()