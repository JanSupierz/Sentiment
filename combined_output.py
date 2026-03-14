# ===== File: Experiment.ipynb =====
import tensorflow as tf
import os
import warnings
import logging
import sys
import yaml
import subprocess
from pathlib import Path
from IPython.display import Image, display

# Set PYTHONPATH to project root so we can import 'src'
project_root = os.getcwd()
os.environ["PYTHONPATH"] = project_root
print(f"PYTHONPATH set to: {project_root}")

!python -m spacy download en_core_web_sm

subprocess.run(["python", "-m", "scripts.loader"], check=True)

import pandas as pd


def preview_parquet_compact(filepath, n=2, text_chars=200, token_items=30):
    df = pd.read_parquet(filepath)

    print(f"File: {filepath}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    df_sample = df.head(n)

    for col in df_sample.columns:
        print("=" * 60)
        print(f"COLUMN: {col}")
        print("=" * 60)

        for i, val in enumerate(df_sample[col]):
            print(f"--- Row {i} ---")

            # Handle long text
            if isinstance(val, str):
                preview = val[:text_chars]
                if len(val) > text_chars:
                    preview += " ..."
                print(preview)

            # Handle token arrays / lists
            elif isinstance(val, (list, tuple)) or hasattr(val, "__array__"):
                tokens = list(val)
                preview = tokens[:token_items]
                print(" ".join(preview))

                if len(tokens) > token_items:
                    print(f"... ({len(tokens)-token_items} more tokens)")

            else:
                print(val)

            print()


preview_parquet_compact('data/preprocessed/train.parquet', n=2)

!python -m scripts.run_basic

import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Configuration ---
MODELS = {"linear_svm": "Linear SVM", "logreg": "LogReg"}
TEXT_VERSIONS = ["text_clean", "text_expanded"]
CASINGS = ["cased", "lower"]
REPRESENTATIONS = ["BoW", "TF-IDF"]

COLOR_MAP = {"Linear SVM": "#1f77b4", "LogReg": "#ff7f0e"}
LINESTYLE_MAP = {"TF-IDF": "-", "BoW": "--"}


def adjust_y_positions(ys: List[float], threshold: float = 0.0025) -> np.ndarray:
    """Adjust overlapping y-coordinates to prevent text overlap."""
    ys = np.array(ys)
    indices = np.argsort(ys)
    adjusted = ys.copy()

    for _ in range(50):
        for i in range(len(adjusted) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            diff = adjusted[idx2] - adjusted[idx1]
            if diff < threshold:
                push = (threshold - diff) / 2.0
                adjusted[idx1] -= push
                adjusted[idx2] += push
    return adjusted


def load_experiment_data(reports_dir: Path) -> pd.DataFrame:
    """Load CSV reports and pivot them into a structure ready for plotting."""
    records = []

    # itertools.product replaces the 4 nested for-loops
    combinations = itertools.product(
        MODELS.items(), TEXT_VERSIONS, CASINGS, REPRESENTATIONS
    )

    for (model_key, model_label), text_ver, case, rep in combinations:
        filepath = reports_dir / f"{text_ver}_{case}_{rep}_{model_key}.csv"

        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            records.append({
                "Pipeline": f"{model_label} ({rep})",
                "Model": model_label,
                "Representation": rep,
                "Casing": case,
                "Text Version": text_ver,
                "F1": df.loc["macro avg", "f1-score"]
            })

    if not records:
        return pd.DataFrame()

    results_df = pd.DataFrame(records)
    pivot_df = results_df.pivot(
        index=["Pipeline", "Model", "Representation", "Casing"],
        columns="Text Version",
        values="F1"
    ).reset_index().dropna(subset=["text_clean", "text_expanded"])

    return pivot_df


def create_slopegraph(df: pd.DataFrame):
    """Generate and display the slopegraph from the pivoted dataframe."""
    cased_df = df[df["Casing"] == "cased"].copy()
    lower_df = df[df["Casing"] == "lower"].copy()

    # Calculate global Y limits for shared axes
    all_f1 = pd.concat([
        cased_df["text_clean"], cased_df["text_expanded"],
        lower_df["text_clean"], lower_df["text_expanded"]
    ])
    padding = 0.05 * (all_f1.max() - all_f1.min())
    y_limits = (all_f1.min() - padding, all_f1.max() + padding)

    # Plot setup
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor='white')
    titles = ["Cased", "Lower (uncased)"]
    data_frames = [cased_df, lower_df]

    for ax, ax_df, title in zip(axes, data_frames, titles):
        _setup_axis(ax, title, y_limits)
        _draw_slope_lines(ax, ax_df)

    _add_legend_and_titles(fig)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()
    plt.close(fig)


def _setup_axis(ax: plt.Axes, title: str, y_limits: tuple):
    """Helper to format individual subplots."""
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean Text", "Expanded Text"], fontsize=11, fontweight='bold', color="#444444")
    ax.tick_params(axis='x', length=0, pad=10)
    ax.tick_params(axis='y', left=False) 
    ax.set_ylim(y_limits)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    # Draw vertical guide lines
    ax.axvline(0, color='#E0E0E0', lw=1.5, zorder=1)
    ax.axvline(1, color='#E0E0E0', lw=1.5, zorder=1)


def _draw_slope_lines(ax: plt.Axes, df: pd.DataFrame):
    """Draw the lines, scatter points, and labels for a single subplot."""
    clean_vals = df["text_clean"].tolist()
    expanded_vals = df["text_expanded"].tolist()

    # Calculate exact text positioning
    clean_adj = adjust_y_positions(clean_vals)
    expanded_adj = adjust_y_positions(expanded_vals)

    for i, (_, row) in enumerate(df.iterrows()):
        y_clean = row["text_clean"]
        y_exp = row["text_expanded"]
        c = COLOR_MAP[row["Model"]]
        ls = LINESTYLE_MAP[row["Representation"]]

        # Main line and markers
        ax.plot([0, 1], [y_clean, y_exp], color=c, linestyle=ls, lw=2.5, alpha=0.85, zorder=2)
        ax.scatter([0, 1], [y_clean, y_exp], color=c, s=70, edgecolor='white', linewidth=1.5, zorder=4)

        # Left label (clean)
        if abs(clean_adj[i] - y_clean) > 0.0001:
            ax.plot([0, -0.05], [y_clean, clean_adj[i]], color=c, lw=1, alpha=0.4, zorder=1)
        ax.text(-0.06, clean_adj[i], f"{y_clean:.4f}", ha='right', va='center',
                fontsize=10, color='#333333', fontweight='500', zorder=5)

        # Right label (expanded)
        if abs(expanded_adj[i] - y_exp) > 0.0001:
            ax.plot([1, 1.05], [y_exp, expanded_adj[i]], color=c, lw=1, alpha=0.4, zorder=1)
        ax.text(1.06, expanded_adj[i], f"{y_exp:.4f}", ha='left', va='center',
                fontsize=10, color='#333333', fontweight='500', zorder=5)


def _add_legend_and_titles(fig: plt.Figure):
    """Helper to add the global legend and figure titles."""
    fig.axes[0].set_ylabel("Macro F1 Score", fontsize=12, labelpad=15, fontweight='bold', color="#444444")

    legend_elements = [
        Line2D([0], [0], color=COLOR_MAP['Linear SVM'], lw=3, label='Linear SVM'),
        Line2D([0], [0], color=COLOR_MAP['LogReg'], lw=3, label='Logistic Regression'),
        Line2D([], [], color='none', label='   '),  # Invisible spacer
        Line2D([0], [0], color='#555555', lw=2.5, linestyle='-', label='TF-IDF'),
        Line2D([0], [0], color='#555555', lw=2.5, linestyle='--', label='BoW')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02), columnspacing=1.5)

    fig.suptitle("Effect of Text Expansion on Macro F1 Score", fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.92,
             "Tracking model performance shifts by algorithm (color) and feature representation (line style).",
             ha='center', fontsize=12, color='#555555')


if __name__ == "__main__":
    target_dir = Path("results/val/run_basic/classification_reports")

    if not target_dir.exists():
        print(f"Could not find directory: {target_dir}")
    else:
        pivot_data = load_experiment_data(target_dir)
        if pivot_data.empty:
            print("No matching experiment files found!")
        else:
            create_slopegraph(pivot_data)

!python -m scripts.run_on_preprocessing

!python -m scripts.run_custom

!python -m scripts.visualize_custom_filtering --column tokens_lower
display(Image("results/figures/analysis/tokens_lower_wordclouds.png"))

!python -m scripts.visualize_custom_filtering --column tokens_filtered
display(Image("results/figures/analysis/tokens_filtered_wordclouds.png"))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import gc

def plot_zscore_experiment_polished(reports_dir="results/val/run_custom/classification_reports"):
    reports_path = Path(reports_dir)

    if not reports_path.exists():
        print(f"Could not find directory: {reports_path}")
        return

    # Configuration
    token_columns = [
        'tokens_cased', 'tokens_lower', 'tokens_letters',
        'tokens_filtered', 'tokens_stemmed', 'tokens_lemmatized'
    ]

    models = {"linear_svm": "Linear SVM", "logreg": "Logistic Regression"}
    variants = ["CustomBow", "CustomTfidf"]

    z_scores = [0, 1, 2, 3, 4]
    records = []

    # Extract Data
    for token_col in token_columns:
        for variant in variants:
            for z in z_scores:
                for model_key, model_label in models.items():

                    filename = f"{token_col}_{variant}_Z{z}_{model_key}.csv"
                    filepath = reports_path / filename

                    if filepath.exists():
                        df = pd.read_csv(filepath, index_col=0)

                        records.append({
                            "Token Processing": token_col.replace('tokens_', '').title(),
                            "Variant": variant,
                            "Z-Score Threshold": float(z),
                            "Model": model_label,
                            "Macro F1-Score": df.loc["macro avg", "f1-score"]
                        })

    if not records:
        print("No matching experiment files found!")
        return

    results_df = pd.DataFrame(records)

    # Find best overall configuration for the printout
    best_run = results_df.loc[results_df['Macro F1-Score'].idxmax()]

    print("\nBEST CONFIGURATION")
    print("-" * 40)
    print(f"Top Model:           {best_run['Model']}")
    print(f"Feature Variant:     {best_run['Variant']}")
    print(f"Top Preprocessing:   {best_run['Token Processing']}")
    print(f"Optimal Z-Score:     {best_run['Z-Score Threshold']}")
    print(f"Peak Macro F1-Score: {best_run['Macro F1-Score']:.4f}")
    print("-" * 40)

    print("\nTop 5 Runs Overall:")
    display(results_df.sort_values(by='Macro F1-Score', ascending=False).head(5))

    plt.rcParams['font.family'] = 'sans-serif'

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 10),
        sharey=True,
        facecolor='#F8F9FA'
    )

    fig.patch.set_facecolor('#F8F9FA')

    bold_palette = {
        'Cased': '#2c3e50',      # Dark Navy
        'Lower': '#e74c3c',      # Red
        'Letters': '#27ae60',    # Green
        'Filtered': '#8e44ad',   # Purple
        'Stemmed': '#f39c12',    # Orange
        'Lemmatized': '#00CCCC'  # Aqua
    }

    axes = axes.flatten()
    plot_index = 0

    # Draw Subplots
    for variant in variants:
        for model_key, model_label in models.items():

            ax = axes[plot_index]
            plot_index += 1

            ax.set_facecolor('white')

            model_df = results_df[
                (results_df["Model"] == model_label) &
                (results_df["Variant"] == variant)
            ]

            if not model_df.empty:
                # Find the Z-score that produced the highest F1
                best_z_for_subplot = model_df.loc[model_df['Macro F1-Score'].idxmax()]['Z-Score Threshold']
                ax.axvline(best_z_for_subplot, color='#333333', linestyle=':', linewidth=2, zorder=1, alpha=0.6)
                ax.axvspan(0, best_z_for_subplot, color='#2ecc71', alpha=0.05, zorder=0)

                x_offset = 0.05 if best_z_for_subplot == 0 else -0.05
                h_align = 'left' if best_z_for_subplot == 0 else 'right'

                ax.text(
                    best_z_for_subplot + x_offset,  
                    0.05,
                    f'Optimal (Z={int(best_z_for_subplot)})',
                    color='#333333',
                    fontsize=11,
                    fontweight='bold',
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    ha=h_align,
                    va='bottom'
                )

            sns.lineplot(
                data=model_df,
                x="Z-Score Threshold",
                y="Macro F1-Score",
                hue="Token Processing",
                palette=bold_palette,
                marker="o",
                markersize=7,
                linewidth=2.5,
                ax=ax,
                legend=(plot_index == 1),
                zorder=3
            )

            # Titles & Axes Styling
            ax.set_title(
                f"{model_label} ({variant})",
                fontsize=15,
                fontweight='bold',
                pad=15
            )

            ax.set_xlim(-0.2, 4.2)
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=11, fontweight='bold')
            ax.set_xlabel("Z-Score Pruning Threshold", fontsize=12, fontweight='bold', labelpad=10)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            ax.spines['left'].set_color('#DDDDDD')
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.grid(axis='x', visible=False)

    # Global labels
    axes[0].set_ylabel("Macro F1-Score", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Macro F1-Score", fontsize=13, fontweight='bold')

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=6,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
        title="Tokenization Strategy",
        title_fontproperties={'weight': 'bold'}
    )

    # Global Titles
    fig.suptitle(
        "Impact of Extreme Vocabulary Pruning on Performance",
        fontsize=18,
        fontweight='bold',
        y=1.02
    )

    fig.text(
        0.5,
        0.96,
        "Green shaded region represents the optimal threshold before performance degradation.",
        ha='center',
        fontsize=12,
        color='#555555'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "zscore_threshold.png"

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved polished plot to: {out_path}")

    plt.show()

    # Cleanup
    plt.close(fig)
    del records, results_df, fig, axes
    gc.collect()

if __name__ == "__main__":
    plot_zscore_experiment_polished()

!python -m scripts.visualize_ablation_study
display(Image("results/figures/analysis/ablation_study.png"))

!python -m scripts.run_n_grams

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ngram_results_polished(reports_dir="results/val/run_n_grams/classification_reports"):
    reports_path = Path(reports_dir)

    ngram_ranges = ["1-1", "1-2", "1-3"]
    models = {"linear_svm": "Linear SVM", "logreg": "Logistic Regression"}
    token_col = "tokens_lower"
    z_threshold = 2.0

    records = []

    # 1. Load data
    for ng in ngram_ranges:
        for m_key, m_label in models.items():
            filename = f"ng{ng}_{m_key}.csv"
            filepath = reports_path / filename

            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0)
                records.append({
                    "N-Gram Range": f"1 to {ng.split('-')[1]}", 
                    "Model": m_label,
                    "Macro F1-Score": df.loc["macro avg", "f1-score"]
                })

    if not records:
        print("No matching files found!")
        return

    results_df = pd.DataFrame(records)

    # 2. Canvas Setup
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
    ax.set_facecolor('white')

    custom_palette = {"Linear SVM": "#2c3e50", "Logistic Regression": "#e74c3c"}

    # 3. Draw Bar Plot (Grouped by N-Gram Range)
    sns.barplot(
        data=results_df, 
        x="N-Gram Range", 
        y="Macro F1-Score", 
        hue="Model",
        palette=custom_palette,
        edgecolor='white',
        linewidth=1.5,
        ax=ax,
        zorder=3,
        alpha=0.9
    )

    # 4. Annotate Bars with exact values
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0: # Avoid annotating empty/ghost bars if any exist
            # Check if this is the absolute best bar to bold it
            is_best = (height == results_df['Macro F1-Score'].max())
            weight = 'heavy' if is_best else 'bold'
            color = '#111111' if is_best else '#444444'
            prefix = "★ " if is_best else ""

            ax.text(
                patch.get_x() + patch.get_width() / 2, 
                height - 0.002, # Tuck the text slightly inside the top of the bar
                f"{prefix}{height:.4f}",
                ha='center', 
                va='top', 
                fontsize=11, 
                color='white', # White text inside the dark bars looks super clean
                fontweight='bold',
                zorder=5
            )

    # 5. Formatting & Cleanup
    ax.set_title(
        f"Impact of N-Gram Ranges on Model Performance (Z={z_threshold})", 
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_ylabel("Macro F1-Score", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel("Vocabulary Depth (N-Grams)", fontsize=12, fontweight='bold', labelpad=10)

    # Clean Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Gridlines behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # Set Y-axis to start at a reasonable baseline so differences are visible but not exaggerated
    y_min = 0.88 
    y_max = results_df["Macro F1-Score"].max() + 0.005
    ax.set_ylim(y_min, y_max)

    # Legend placement
    ax.legend(
        title="Algorithm", 
        title_fontproperties={'weight': 'bold'},
        loc='upper left', # Move it out of the way of the taller right-side bars
        frameon=True,
        edgecolor='#DDDDDD',
        facecolor='white'
    )

    plt.tight_layout()

    # 6. Save and show
    out_dir = Path("results/figures/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ngram_experiment_bars.png"
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved polished grouped bar chart to: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_ngram_results_polished()

!python -m scripts.visualize_certainty
display(Image("results/figures/analysis/model_comparison_certainty_dist.png"))

config_path = "configs/default.yaml"

with open(config_path) as f:
    base_cfg = yaml.safe_load(f)

print(f"Keys: {base_cfg}")

# 2. Build Feature Factory
!python -m scripts.build_features --config configs/default.yaml

!python -m scripts.visualize_concepts --token_col tokens_lower --n_concepts 10000 --weights 0 10 --top_n 10

display(Image("results/figures/concepts/pos_wc_tokens_lower_k10000_w0.png"))
display(Image("results/figures/concepts/pos_wc_tokens_lower_k10000_w10.png"))
display(Image("results/figures/concepts/neg_wc_tokens_lower_k10000_w0.png"))
display(Image("results/figures/concepts/neg_wc_tokens_lower_k10000_w10.png"))

!python -m scripts.grid_search --config configs/default.yaml --workers 1

!python -m scripts.visualize_cluster_comparison
display(Image("results/figures/analysis/cluster_comparison.png"))

print("\n--- Model Comparison: Baseline vs Clustered Concepts ---")
!python -m scripts.compare_models
display(Image("results/figures/analysis/champion_confusion_matrices.png"))

!python -m scripts.train_bert --config configs/default.yaml

!python -m scripts.analyse_results
display(Image("results/figures/analysis/accuracy_coverage_tradeoff.png"))

print("--- Step 6: Train Specialist BERT (Transfer Learning) ---")
!python -m scripts.train_specialist --config configs/default.yaml --best_params configs/best_params.yaml

print("--- Step 7: Run Ensemble (Cascade Evaluation) ---")
!python -m scripts.run_ensemble --config configs/default.yaml --best_params configs/best_params.yaml

!python -m scripts.visualize_thesis
display(Image("results/figures/thesis/thesis_final_results.png"))




# ===== File: .virtual_documents/Experiment.ipynb =====


# ===== File: configs/best_params.yaml =====
cascade:
  delegation_threshold: 0.244
features:
  n_concepts: 0
  ngram_range:
  - 1
  - 3
  sentiment_weight: 0.0
  use_concepts: false
  z_threshold: 1.96
model: linear_svm
models:
  linear_svm:
    C: 1.0


# ===== File: configs/default.yaml =====
# ==========================================
# STATIC PIPELINE SETTINGS
# ==========================================
bert:
    sentence_model: "all-MiniLM-L6-v2"
    basic:
        model_name: "distilbert-base-uncased-finetuned-sst-2-english"
        epochs: 3
        batch_size: 16
        max_len: 256
        learning_rate: 2e-5
        patience: 2
    specialist:
        epochs: 3
        batch_size: 8
        learning_rate: 1e-5
        num_layers_to_freeze: 4
        patience: 8

# ==========================================
# GRID SEARCH SPACE (Iterated by Dask)
# ==========================================
grid_search:
    models: ["linear_svm", "logreg"]
    n_concepts: [0, 500, 5000, 10000]
    sentiment_weight: [0.0, 1.0, 10.0]

# ===== File: scripts/analyse_results.py =====
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src.utils.paths import RESULTS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


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

        # LOCAL ACCURACY (What the SVM scores on its retained portion)
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

    # FIND OVERALL WINNER (Based on max Hybrid Accuracy)
    best_row = curve_df.loc[curve_df['Hybrid_Accuracy'].idxmax()]

    winner_name = best_row['Model']
    best_certainty = best_row['Certainty_Threshold']
    best_delegated = best_row['Delegated']
    best_hybrid_acc = best_row['Hybrid_Accuracy']
    lower_bound = 1.0 - best_certainty

    print(f"\nCASCADE OPTIMIZATION WINNER: {winner_name}")
    print(f"-> Peak Global System Accuracy: {best_hybrid_acc:.2%} (Beats Base BERT by {best_hybrid_acc - bert_global_acc:+.2%})")
    print(f"-> Required Model Certainty: {best_certainty:.1%} (Accepts probs >= {best_certainty:.3f} or <= {lower_bound:.3f})")
    print(f"-> Data Delegated to BERT: {best_delegated:.2%} (SVM handles {(1-best_delegated):.2%})")

# PLOTTING GRAPHS
    top_svms = curve_df.groupby('Model')['Hybrid_Accuracy'].max().sort_values(ascending=False).head(4).index.tolist()
    plot_df = curve_df[curve_df['Model'].isin(top_svms)].copy()
    plot_df['Model'] = plot_df['Model'].apply(lambda x: x.replace("linear_svm", "SVM").replace("logreg", "LR"))
    clean_top_svms = [m.replace("linear_svm", "SVM").replace("logreg", "LR") for m in top_svms]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Cascade Pipeline Analysis & Global System Optimization", fontsize=20, fontweight='bold', y=1.05)

    colors = sns.color_palette("tab10", len(clean_top_svms))
    palette = {m: c for m, c in zip(clean_top_svms, colors)}

    # --- Graph 1: Local Accuracy vs Certainty Requirement ---
    # ADDED: style="Model", alpha=0.8 to make overlapping lines distinct
    sns.lineplot(data=plot_df, x="Certainty_Threshold", y="Local_Accuracy", hue="Model", 
                 style="Model", dashes=True, alpha=0.8,
                 palette=palette, ax=ax1, linewidth=2.5, legend=False)
    ax1.set_title("1. Local Accuracy (SVM on Retained Data)", fontsize=14)
    ax1.set_xlabel(r"Model Prediction Certainty (%)", fontsize=12)
    ax1.set_ylabel("Local Accuracy (%)", fontsize=12)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_xlim(0.5, 1.0) 
    ymin, _ = ax1.get_ylim()
    ax1.set_ylim(ymin, 1.0)

    # --- Graph 2: Deferral Curve (Data Delegated vs Certainty) ---
    # ADDED: style="Model", alpha=0.8
    sns.lineplot(data=plot_df, x="Certainty_Threshold", y="Delegated", hue="Model", 
                 style="Model", dashes=True, alpha=0.8,
                 palette=palette, ax=ax2, linewidth=2.5, legend=False)
    ax2.set_title("2. Workload Management (Deferral Curve)", fontsize=14)
    ax2.set_xlabel(r"Model Prediction Certainty (%)", fontsize=12)
    ax2.set_ylabel("Data Delegated to BERT (%)", fontsize=12)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(0.0, 1.0)

    # --- Graph 3: True Hybrid Optimization Score ---
    # ADDED: style="Model", alpha=0.8
    sns.lineplot(data=plot_df, x="Delegated", y="Hybrid_Accuracy", hue="Model", 
                 style="Model", dashes=True, alpha=0.8,
                 palette=palette, ax=ax3, linewidth=2.5)

    ax3.axhline(bert_global_acc, color='black', linestyle='--', linewidth=2, label=f"BERT Global Baseline ({bert_global_acc:.1%})")
    ax3.plot(best_delegated, best_hybrid_acc, marker='*', markersize=18, color='gold', markeredgecolor='black', zorder=10, label="Optimal Peak")

    ax3.set_title("3. Global System Optimization Score", fontsize=14)
    ax3.set_xlabel("Data Delegated to BERT (%)", fontsize=12)
    ax3.set_ylabel("Combined Cascade Accuracy (%)", fontsize=12)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Push legend outside or adjust so styles are clearly visible
    ax3.legend(fontsize='11', loc='upper right', bbox_to_anchor=(1.0, 0.95))
    ax3.set_xlim(0.0, 1.0)

    plt.tight_layout()
    plot_path = fig_dir / "accuracy_coverage_tradeoff.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved Trade-off graphs to: {plot_path}")

    try:
        from IPython.display import display, Image
        display(Image(filename=plot_path))
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep cascade thresholds across multiple experiment runs.")
    parser.add_argument("--experiments", nargs="+", default=["grid_search", "run_n_grams", "run_basic", "run_custom", "run_on_preprocessing"], 
                        help="List of experiment folders to pull SVM/LogReg CSVs from (e.g. grid_search run_n_grams)")
    parser.add_argument("--bert_exp", default="bert",
                        help="The experiment folder containing your bert_basic_baseline.csv")

    args = parser.parse_args()
    main(args.experiments, args.bert_exp)

# ===== File: scripts/build_features.py =====
import argparse
import yaml
from src.utils.paths import PROJECT_ROOT
from src.features import builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Get parameter spaces from config
    grid = cfg['grid_search']
    n_concepts_list = grid['n_concepts']
    sentiment_weights = grid['sentiment_weight']

    print("=== PHASE 1: Build unit matrices ===")
    builder.build_unit_matrices('tokens_lower', max_df=0.7)

    print("\n=== PHASE 2: Compute unit Z‑score indices ===")
    builder.compute_unit_z_indices('tokens_lower', [2.0])

    print("\n=== PHASE 3: Compute embeddings (needed for concepts) ===")
    builder.compute_embeddings('tokens_lower')

    print("\n=== PHASE 4: Extract concepts for each (k, w) combination ===")
    for nc in n_concepts_list:
        if nc == 0:
            continue
        for w in sentiment_weights:
            builder.extract_concepts('tokens_lower', nc, w)

    print("\n=== PHASE 5: Build concept matrices ===")
    for nc in n_concepts_list:
        if nc == 0:
            continue
        for w in sentiment_weights:
            builder.build_concept_matrices('tokens_lower', nc, w)

    print("\n=== PHASE 6: Compute concept Z‑score indices ===")
    for nc in n_concepts_list:
        if nc == 0:
            continue
        for w in sentiment_weights:
            builder.compute_concept_z_indices('tokens_lower', nc, w, [2.0])


if __name__ == "__main__":
    main()

# ===== File: scripts/compare_models.py =====
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils.paths import RESULTS_DIR, FIGURES_DIR

# Set a clean academic theme
sns.set_theme(style="white", context="paper", font_scale=1.2)

def main():
    raw_dir = RESULTS_DIR / "val" / "raw_predictions"
    if not raw_dir.exists():
        print("❌ No raw predictions found. Run grid search first.")
        return

    results = []
    
    # 1. Parse all CSV files and calculate accuracy
    for f in raw_dir.glob("*.csv"):
        # Skip BERT baselines, we only want the classic models for this comparison
        if f.name == "bert_basic_baseline.csv" or "bert_specialist" in f.name:
            continue
            
        stem = f.stem
        model_type = "SVM" if "linear_svm" in stem else "LogReg"
        # If 'k0' is in the filename, it's a baseline (non-clustered)
        is_clustered = "k0" not in stem
        
        df = pd.read_csv(f)
        preds = (df['probability'] > 0.5).astype(int)
        acc = (preds == df['true_label']).mean()
        
        results.append({
            "Model_Type": model_type,
            "Is_Clustered": is_clustered,
            "Config": stem,
            "Accuracy": acc,
            "Predictions": preds.values,
            "True_Labels": df['true_label'].values
        })

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("❌ Not enough data to compare.")
        return

    # 2. Define our 4 categories and find the champions
    champions = []
    categories = [
        ("SVM", False, "Best Baseline SVM (Non-Clustered)"),
        ("SVM", True, "Best Concept SVM (Clustered)"),
        ("LogReg", False, "Best Baseline LogReg (Non-Clustered)"),
        ("LogReg", True, "Best Concept LogReg (Clustered)")
    ]

    print("\n🏆 === CHAMPION MODELS SUMMARY === 🏆")
    for m_type, is_clust, title in categories:
        subset = df_results[(df_results['Model_Type'] == m_type) & (df_results['Is_Clustered'] == is_clust)]
        
        if not subset.empty:
            # Find the row with the maximum accuracy in this category
            best_idx = subset['Accuracy'].idxmax()
            best_row = subset.loc[best_idx]
            champions.append((title, best_row))
            
            print(f"{title}:")
            print(f"  -> Config: {best_row['Config']}")
            print(f"  -> Accuracy: {best_row['Accuracy']:.4f}\n")

    if len(champions) < 4:
        print("⚠️ Warning: Did not find models for all 4 categories. Proceeding with what was found.")

    # 3. Plot the 2x2 Confusion Matrix Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Confusion Matrices: Baseline vs. Clustered Concept Models", fontsize=18, fontweight='bold', y=0.98)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    for i, (title, row) in enumerate(champions):
        ax = axes[i]
        cm = confusion_matrix(row['True_Labels'], row['Predictions'])
        
        # Draw the heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    annot_kws={"size": 14}, ax=ax)
        
        ax.set_title(f"{title}\nAcc: {row['Accuracy']:.2%}", fontsize=13, pad=10)
        ax.set_ylabel('Actual Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
        ax.set_yticklabels(['Negative (0)', 'Positive (1)'])

    # Hide any unused subplots if we found less than 4 champions
    for j in range(len(champions), 4):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_path = FIGURES_DIR / "analysis" / "champion_confusion_matrices.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"📊 Saved confusion matrix grid to: {out_path}")

if __name__ == "__main__":
    main()

# ===== File: scripts/grid_search.py =====
#!/usr/bin/env python3
import argparse
import yaml
import itertools
from sklearn.feature_extraction.text import TfidfTransformer
import dask
from dask.distributed import Client, LocalCluster, as_completed
from tqdm import tqdm

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.utils.paths import PROJECT_ROOT
from src.features import builder


def run_grid_iter(token_col, n_concepts, sentiment_weight, model_name, static_cfg):
    run_id = f"{model_name}_k{n_concepts}_w{int(sentiment_weight)}"

    try:
        X_train, y_train = builder.load_representation(
            token_col, n_concepts, sentiment_weight, 2, 'train'
        )
        X_val, y_val = builder.load_representation(
            token_col, n_concepts, sentiment_weight, 2, 'val'
        )

        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)

        if model_name == 'linear_svm':
            model = LinearSVMClassifier(name=run_id)
        else:
            model = LogisticRegressionClassifier(name=run_id)

        model.train(X_train_tfidf, y_train)
        model.evaluate(X_val_tfidf, y_val, name="val/grid_search")

        return f"Done: {run_id}"
    except Exception as e:
        return f"Failed {run_id}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--token_col", default="tokens_lower", help="Token column to use")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        static_cfg = yaml.safe_load(f)
    grid = static_cfg['grid_search']

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    token_col = args.token_col

    delayed_tasks = []
    for nc, w, m in itertools.product(
        grid['n_concepts'], grid['sentiment_weight'], grid['models']
    ):
        if nc == 0 and w > 0:
            continue
        delayed_tasks.append(dask.delayed(run_grid_iter)(
            token_col, nc, w, m, static_cfg
        ))

    if delayed_tasks:
        print(f"Starting Grid Search with {len(delayed_tasks)} iterations...")
        futures = client.compute(delayed_tasks)

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search Progress"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"Failed: {e}")

        failed = [r for r in results if r.startswith("Failed")]
        print(f"\nCompleted: {len(results) - len(failed)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for f in failed[:5]: print(f"  {f}")

    client.close()


if __name__ == "__main__":
    main()

# ===== File: scripts/loader.py =====
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any

import contractions
import nltk
import pandas as pd
import spacy
import tensorflow_datasets as tfds
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def get_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    """Generate n-grams from a list of tokens."""
    min_n, max_n = ngram_range
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


TRANSLATE_TABLE = str.maketrans({
    "`": "'", "´": "'", "’": "'", "‘": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "\x96": "-", "\x97": "-",
    "…": "..."
})


def normalize_text(text: str) -> str:
    """Unify punctuation, normalise Unicode, strip HTML and extra spaces."""
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(TRANSLATE_TABLE)
    text = re.sub(r'<[^>]+>', ' ', text)         # remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_letters(tokens: List[str]) -> List[str]:
    """Remove all non letter characters from tokens, discard if empty."""
    cleaned = []
    for t in tokens:
        token_clean = re.sub(r'[^a-zA-Z]', '', t)
        if token_clean:
            cleaned.append(token_clean)
    return cleaned


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply Porter stemming to each token."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Apply WordNet lemmatization to each token."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def process_one(item: Tuple[bytes, int]) -> Dict[str, Any]:
    """Clean a single review and produce all text representations."""
    text_raw, label = item
    if isinstance(text_raw, bytes):
        text_raw = text_raw.decode('utf-8')

    text_clean = normalize_text(text_raw)
    text_expanded = contractions.fix(text_clean)

    tokens_cased = word_tokenize(text_expanded)
    tokens_lower = [w.lower() for w in tokens_cased]

    doc = nlp(text_expanded)
    tokens_filtered = [
        token.text.lower() for token in doc
        if (token.pos_ != 'AUX' or token.tag_ == 'MD')
        and token.pos_ != 'DET'
        and token.tag_ != 'POS'
    ]

    return {
        'sentiment': int(label),
        'text_raw': text_raw,
        'text_clean': text_clean,
        'text_expanded': text_expanded,
        'tokens_cased': tokens_cased,
        'tokens_lower': tokens_lower,
        'tokens_letters': clean_letters(tokens_lower),
        'tokens_stemmed': stem_tokens(tokens_lower),
        'tokens_lemmatized': lemmatize_tokens(tokens_lower),
        'tokens_filtered': tokens_filtered
    }


def load_and_process(n_jobs: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    print("Loading IMDb reviews...")
    ds = tfds.load('imdb_reviews', split='train+test', as_supervised=True)
    raw_data = list(tfds.as_numpy(ds))

    # Deduplicate based on raw review text
    seen = set()
    unique_raw = []
    for t, l in raw_data:
        if t not in seen:
            seen.add(t)
            unique_raw.append((t, l))
    print(f"Removed {len(raw_data)-len(unique_raw)} duplicates.")

    print(f"Processing {len(unique_raw)} reviews...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        processed = []
        for result in tqdm(executor.map(process_one, unique_raw), total=len(unique_raw)):
            processed.append(result)

    labels = [item['sentiment'] for item in processed]

    train, temp, _, temp_labels = train_test_split(
        processed, labels, test_size=0.5,
        random_state=42, stratify=labels
    )

    val, test = train_test_split(
        temp, test_size=0.3,
        random_state=42, stratify=temp_labels
    )

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def save_as_parquet(train, val, test, out_dir):
    """Convert lists to DataFrames and save as Parquet."""
    os.makedirs(out_dir, exist_ok=True)

    for name, data in zip(['train', 'val', 'test'], [train, val, test]):
        df = pd.DataFrame(data)

        df = df[['sentiment', 'text_raw', 'text_clean', 'text_expanded',
                 'tokens_cased', "tokens_lower", 'tokens_filtered',
                 'tokens_letters', 'tokens_stemmed', 'tokens_lemmatized']]

        out_path = os.path.join(out_dir, f'{name}.parquet')
        df.to_parquet(out_path, index=False)
        print(f"Saved {name} set to {out_path}")


if __name__ == '__main__':
    save_as_parquet(*load_and_process(n_jobs=4), out_dir='data/preprocessed')

# ===== File: scripts/param_analysis.py =====
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

from src.utils.paths import RESULTS_DIR, FIGURES_DIR

# Set a clean, minimalist theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def load_parsed_results():
    raw_dir = RESULTS_DIR / "val" / "raw_predictions"
    results = []
    
    for f in raw_dir.glob("*.csv"):
        if f.name == "bert_basic_baseline.csv": 
            continue
            
        stem = f.stem
        model = "SVM" if "linear_svm" in stem else "LogReg"
        rest = stem.replace("linear_svm_", "").replace("logreg_", "")
        parts = rest.split('_')
        
        try:
            ng = [p for p in parts if p.startswith('ng')][0].replace('ng', '')
            k = int([p for p in parts if p.startswith('k')][0].replace('k', ''))
            z = float([p for p in parts if p.startswith('z')][0].replace('z', ''))
            w = float([p for p in parts if p.startswith('w')][0].replace('w', ''))
            
            df = pd.read_csv(f)
            acc = ((df['probability'] > 0.5).astype(int) == df['true_label']).mean()
            
            w_label = f"w={int(w)}" if w > 0 else "w=0"
            
            results.append({
                "Model": model,
                "N-Grams": ng,
                "Z-Score": z,
                "k_val": k,
                "VADER (w)": w_label,
                "w_val": w,
                "Accuracy": acc,
                # Create a strict unique ID for every single trajectory
                "Config_ID": f"{model}_{ng}_w{w}"
            })
        except IndexError: 
            continue
            
    return pd.DataFrame(results)

def main():
    df = load_parsed_results()
    if df.empty:
        print("❌ No valid grid search results found.")
        return

    # Sort data to ensure lines draw cleanly left-to-right
    df = df.sort_values(by=["k_val", "N-Grams", "Model", "w_val", "Z-Score"])

    unique_k = sorted(df["k_val"].unique())
    ng_order = sorted(df["N-Grams"].unique())
    w_order_vals = sorted(df["w_val"].unique())
    w_order = [f"w={int(w)}" if w > 0 else "w=0" for w in w_order_vals]
    
    n_plots = len(unique_k)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots), sharex=True)
    fig.suptitle("Impact of Z-Score Pruning Broken Down by Granularity & N-Grams", fontsize=20, fontweight='bold', y=0.98)
    
    if n_plots == 1:
        axes = [axes]

    ng_palette = {"1-1": "#2ecc71", "1-2": "#3498db", "1-3": "#9b59b6"}
    model_markers = {"SVM": "o", "LogReg": "s"}

    for ax, k in zip(axes, unique_k):
        subset = df[df["k_val"] == k].copy()
        
        title = "Baseline (No Clusters)" if k == 0 else f"Clustered: {k} Concepts"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        
        # 1. DRAW THE LINES 
        # By using units="Config_ID" and estimator=None, we force a dedicated line 
        # for every single distinct configuration (no shade, no averaging)
        sns.lineplot(
            data=subset, x="Z-Score", y="Accuracy", 
            hue="N-Grams", palette=ng_palette, hue_order=ng_order,
            style="Model", 
            units="Config_ID", estimator=None, 
            markers=False, dashes=True, linewidth=1.5, 
            ax=ax, legend=True, alpha=0.5
        )

        # 2. DRAW THE DOTS
        sns.scatterplot(
            data=subset, x="Z-Score", y="Accuracy", 
            hue="N-Grams", palette=ng_palette, hue_order=ng_order,
            style="Model", markers=model_markers,
            size="VADER (w)", size_order=w_order, sizes=(250, 50),
            ax=ax, legend="brief", alpha=0.9, edgecolor='black', linewidth=1
        )

        ax.set_ylabel("Global Accuracy", fontsize=13)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Clean up subplot legends so only the top one has the master legend
        if ax != axes[0]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            handles, labels = ax.get_legend_handles_labels()
            
            clean_handles, clean_labels = [], []
            seen_labels = set() # Track what we've added to avoid duplicates
            
            for h, l in zip(handles, labels):
                # Remove messy internal trajectory IDs
                if l == "Config_ID" or l.startswith("SVM_") or l.startswith("LogReg_"):
                    continue
                
                # Add label if we haven't seen it yet (prevents line/scatter duplication)
                if l not in seen_labels:
                    clean_handles.append(h)
                    clean_labels.append(l)
                    seen_labels.add(l)
                    
            ax.legend(handles=clean_handles, labels=clean_labels, 
                      bbox_to_anchor=(1.02, 1), loc='upper left', 
                      fontsize=11, frameon=True, title="Configuration Guide")

    # Only add the X-axis label to the very bottom graph
    axes[-1].set_xlabel("Z-Score Threshold (Higher = Stricter Pruning)", fontsize=14)
    axes[-1].set_xticks(sorted(df["Z-Score"].unique()))

    out_path = FIGURES_DIR / "analysis" / "zscore_full_trellis.png"
    plt.subplots_adjust(hspace=0.15) 
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved clean trellis graph to: {out_path}")

    try:
        from IPython.display import display, Image
        display(Image(filename=out_path))
    except ImportError:
        pass

if __name__ == "__main__":
    main()

# ===== File: scripts/run_basic.py =====
import pandas as pd
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from tqdm.auto import tqdm


def main():
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    models = ["linear_svm", "logreg"]
    text_columns = ["text_clean", "text_expanded"]
    casing_options = [True, False]

    total_iters = len(text_columns) * len(casing_options) * 2 * len(models)

    with tqdm(total=total_iters, desc="Training & evaluating", ncols=100) as pbar:

        for text_col in text_columns:
            for is_lower in casing_options:
                # Create a label for saving the models/metrics cleanly
                case_label = "lower" if is_lower else "cased"

                vectorizer = CountVectorizer(
                    ngram_range=(1, 3),
                    min_df=10,
                    max_df=0.7,
                    lowercase=is_lower
                )

                # Extract features for the current text column and casing
                X_train = vectorizer.fit_transform(train_df[text_col])
                X_val = vectorizer.transform(val_df[text_col])

                # --- BoW Phase ---
                for model_name in models:
                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=f"{text_col}_{case_label}_BoW_{model_name}")
                    else:
                        clf = LogisticRegressionClassifier(name=f"{text_col}_{case_label}_BoW_{model_name}")

                    clf.train(X_train, y_train)
                    clf.evaluate(X_val, y_val, name="val/run_basic")
                    pbar.update(1)

                # --- TF-IDF Phase ---
                tfidf = TfidfTransformer()
                X_train = tfidf.fit_transform(X_train)
                X_val = tfidf.transform(X_val)
                gc.collect()

                for model_name in models:
                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=f"{text_col}_{case_label}_TF-IDF_{model_name}")
                    else:
                        clf = LogisticRegressionClassifier(name=f"{text_col}_{case_label}_TF-IDF_{model_name}")

                    clf.train(X_train, y_train)
                    clf.evaluate(X_val, y_val, name="val/run_basic")
                    pbar.update(1)

                del X_train, X_val
                gc.collect()


if __name__ == "__main__":
    main()

# ===== File: scripts/run_custom.py =====
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.features.sentiment import SentimentFeatures
from tqdm.auto import tqdm
from src.features.vectorizer import build_count_matrix


def main():
    # Load datasets
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")

    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    # Tokenization variants
    token_columns = [
        "tokens_cased",
        "tokens_lower",
        "tokens_letters",
        "tokens_filtered",
        "tokens_stemmed",
        "tokens_lemmatized"
    ]

    # Models to evaluate
    models = ["linear_svm", "logreg"]
    variants = ["CustomBow", "CustomTfidf"]

    # Z-score thresholds for vocabulary pruning
    z_scores = [0, 1, 2, 3, 4]

    # Total iterations for progress bar
    total_iters = len(token_columns) * len(z_scores) * len(models) * 2

    with tqdm(total=total_iters, desc="Feature Variants", ncols=100) as pbar:

        for token_col in token_columns:

            # Build base Count matrix once per tokenization
            X_train_base, X_val_base, _ = build_count_matrix(
                train_df[token_col],
                val_df[token_col],
            )

            # Fit sentiment feature statistics once
            sf = SentimentFeatures()
            sf.fit(X_train_base, y_train)

            # Iterate through Z-score pruning thresholds
            for z in z_scores:

                z_keep_indices = list(sf.filter_by_zscore(z))

                if not z_keep_indices:
                    print(f"\nWarning: Z-score {z} pruned all features for {token_col}. Skipping.")
                    pbar.update(len(models) * 2)
                    continue

                if len(z_keep_indices) < 2:
                    print(f"\nWarning: Z-score {z} left too few features for {token_col}. Skipping.")
                    pbar.update(len(models) * 2)
                    continue

                # Apply Z-score mask
                X_train_filtered = X_train_base[:, z_keep_indices]
                X_val_filtered = X_val_base[:, z_keep_indices]

                # Train models on filtered BoW
                for model_name in models:

                    model_save_name = f"{token_col}_{variants[0]}_Z{z}_{model_name}"

                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=model_save_name)
                    else:
                        clf = LogisticRegressionClassifier(name=model_save_name)

                    clf.train(X_train_filtered, y_train)
                    clf.evaluate(X_val_filtered, y_val, name="val/run_custom")

                    pbar.update(1)

                # TF-IDF transformation
                tfidf = TfidfTransformer()
                X_train_tfidf = tfidf.fit_transform(X_train_filtered)
                X_val_tfidf = tfidf.transform(X_val_filtered)

                # Train models on filtered TF-IDF
                for model_name in models:

                    model_save_name = f"{token_col}_{variants[1]}_Z{z}_{model_name}"

                    if model_name == "linear_svm":
                        clf = LinearSVMClassifier(name=model_save_name)
                    else:
                        clf = LogisticRegressionClassifier(name=model_save_name)

                    clf.train(X_train_tfidf, y_train)
                    clf.evaluate(X_val_tfidf, y_val, name="val/run_custom")

                    pbar.update(1)

                del X_train_filtered, X_val_filtered
                del X_train_tfidf, X_val_tfidf
                gc.collect()

            del X_train_base, X_val_base
            gc.collect()


if __name__ == "__main__":
    main()

# ===== File: scripts/run_ensemble.py =====
#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.models.deep import BERTClassifier
from src.utils.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.features.builder import load_representation

logging.getLogger("transformers").setLevel(logging.ERROR)

def safe_binary_probs(probs):
    if probs.ndim == 2:
        return probs[:, 1]
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / args.best_params) as f:
        best = yaml.safe_load(f)

    cfg.setdefault('features', {})
    cfg.setdefault('models', {})
    cfg.setdefault('model', 'linear_svm')
    cfg.setdefault('cascade', {})

    cfg['features'].update(best.get('features', {}))
    cfg['model'] = best.get('model', cfg['model'])
    if 'models' in best:
        cfg['models'].update(best['models'])
    if 'cascade' in best:
        cfg['cascade'].update(best['cascade'])

    threshold = cfg['cascade']['delegation_threshold']

    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']

    # Load test data – will automatically build test matrix if missing
    X_test_sp, y_test = load_representation(cfg, nr, nc, w, z, 'test')
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")
    X_test_txt = df_test['clean_review'].tolist()

    # TF‑IDF
    X_train_sp, _ = load_representation(cfg, nr, nc, w, z, 'train')
    tfidf = TfidfTransformer()
    tfidf.fit(X_train_sp)
    X_test_tfidf = tfidf.transform(X_test_sp)

    # Load models
    svm_path = MODELS_DIR / "svm_base.joblib"
    if not svm_path.exists():
        raise FileNotFoundError("SVM model not found. Run scripts/train_specialized.py first.")
    svm = joblib.load(svm_path)

    bert_basic_path = MODELS_DIR / "bert_basic"
    if not bert_basic_path.exists():
        raise FileNotFoundError("Basic BERT not found. Run scripts/train_bert.py first.")
    bert_basic = BERTClassifier.load(str(bert_basic_path), name="BERT_Basic")

    bert_spec_path = MODELS_DIR / "bert_specialist"
    if not bert_spec_path.exists():
        raise FileNotFoundError("Specialist BERT not found. Run scripts/train_specialized.py first.")
    bert_spec = BERTClassifier.load(str(bert_spec_path), name="BERT_Specialist")

    # Predictions
    probs_svm = safe_binary_probs(svm.predict_proba(X_test_tfidf))
    preds_svm = (probs_svm > 0.5).astype(int)

    probs_basic = safe_binary_probs(bert_basic.predict_proba(X_test_txt))
    preds_basic = (probs_basic > 0.5).astype(int)

    probs_spec = safe_binary_probs(bert_spec.predict_proba(X_test_txt))
    preds_spec = (probs_spec > 0.5).astype(int)

    # Cascade
    lower = threshold
    upper = 1.0 - lower
    uncertain_mask = (probs_svm >= lower) & (probs_svm <= upper)
    delegation_rate = uncertain_mask.mean()

    preds_cascade_basic = preds_svm.copy()
    preds_cascade_basic[uncertain_mask] = preds_basic[uncertain_mask]

    preds_cascade_spec = preds_svm.copy()
    preds_cascade_spec[uncertain_mask] = preds_spec[uncertain_mask]

    # Accuracies
    acc_svm = accuracy_score(y_test, preds_svm)
    acc_basic = accuracy_score(y_test, preds_basic)
    acc_spec = accuracy_score(y_test, preds_spec)
    acc_cascade_basic = accuracy_score(y_test, preds_cascade_basic)
    acc_cascade_spec = accuracy_score(y_test, preds_cascade_spec)

    # Save results
    results = {
        "SVM": acc_svm,
        "Basic BERT": acc_basic,
        "Specialist BERT": acc_spec,
        "SVM+Basic Cascade": acc_cascade_basic,
        "SVM+Specialist Cascade": acc_cascade_spec,
        "Delegation Rate": delegation_rate,
        "Threshold": threshold
    }
    results_df = pd.DataFrame([results])
    out_dir = RESULTS_DIR / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "ensemble_results.csv", index=False)

    print("\n=== CASCADE ENSEMBLE RESULTS ===")
    for name, acc in results.items():
        if "Rate" not in name and "Threshold" not in name:
            print(f"{name:25s}: {acc:.4f}")
    print(f"Delegation rate        : {delegation_rate:.2%}")
    print(f"Threshold used         : {threshold:.3f}")
    print(f"\nResults saved to {out_dir / 'ensemble_results.csv'}")

if __name__ == "__main__":
    main()

# ===== File: scripts/run_n_grams.py =====
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.features.sentiment import SentimentFeatures
from src.features.vectorizer import build_count_matrix
from tqdm.auto import tqdm

def run_ngram_experiment():
    print("Loading data...")
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    # Experiment parameters
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    models = ["linear_svm", "logreg"]
    total_iters = len(ngram_ranges) * len(models)

    with tqdm(total=total_iters, desc="Testing N-Gram Ranges", ncols=100) as pbar:
        for ngram in ngram_ranges:

            # Build Matrix
            X_train, X_val, _ = build_count_matrix(
                train_df['tokens_lower'], val_df['tokens_lower'],
                ngram_range=ngram, max_df=0.7
            )

            # Apply Z-Score Pruning
            sf = SentimentFeatures()
            sf.fit(X_train, y_train.tolist())
            z_keep_indices = list(sf.filter_by_zscore(2.0))

            X_train = X_train[:, z_keep_indices]
            X_val = X_val[:, z_keep_indices]

            # TF-IDF Transformation
            tfidf = TfidfTransformer()
            X_train = tfidf.fit_transform(X_train)
            X_val = tfidf.transform(X_val)

            # Train & Evaluate
            for model_key in models:
                model_name = f"ng{ngram[0]}-{ngram[1]}_{model_key}"

                if model_key == "linear_svm":
                    clf = LinearSVMClassifier(name=model_name)
                else:
                    clf = LogisticRegressionClassifier(name=model_name)

                clf.train(X_train, y_train)
                clf.evaluate(X_val, y_val, name="val/run_n_grams")
                pbar.update(1)

            # Cleanup memory
            del X_train, X_val
            gc.collect()


if __name__ == "__main__":
    run_ngram_experiment()

# ===== File: scripts/run_on_preprocessing.py =====
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from tqdm.auto import tqdm
from src.features.vectorizer import build_count_matrix


def main():
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    val_df = pd.read_parquet("data/preprocessed/val.parquet")
    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values

    token_columns = [
        'tokens_cased', 'tokens_lower', 'tokens_letters', 'tokens_filtered',
        'tokens_stemmed', 'tokens_lemmatized'
    ]
    models = ["linear_svm", "logreg"]

    # 2 represents BoW and TF-IDF phases
    total_iters = len(token_columns) * 2 * len(models)

    with tqdm(total=total_iters, desc="Training & evaluating", ncols=100) as pbar:
        for token_col in token_columns:

            X_train, X_val, _ = build_count_matrix(train_df[token_col], val_df[token_col])

            for model_name in models:
                if model_name == "linear_svm":
                    clf = LinearSVMClassifier(name=f"{token_col}_BoW_{model_name}")
                else:
                    clf = LogisticRegressionClassifier(name=f"{token_col}_BoW_{model_name}")

                clf.train(X_train, y_train)
                clf.evaluate(X_val, y_val, name="val/run_on_preprocessing")
                pbar.update(1)

            tfidf = TfidfTransformer()
            X_train = tfidf.fit_transform(X_train)
            X_val = tfidf.transform(X_val)
            gc.collect()

            for model_name in models:
                if model_name == "linear_svm":
                    clf = LinearSVMClassifier(name=f"{token_col}_TF-IDF_{model_name}")
                else:
                    clf = LogisticRegressionClassifier(name=f"{token_col}_TF-IDF_{model_name}")

                clf.train(X_train, y_train)
                clf.evaluate(X_val, y_val, name="val/run_on_preprocessing")
                pbar.update(1)

            del X_train, X_val
            gc.collect()


if __name__ == "__main__":
    main()

# ===== File: scripts/train_bert.py =====
#!/usr/bin/env python3
import argparse
import yaml
import pandas as pd
import numpy as np
from src.models.deep import BERTClassifier
from src.utils.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print("\n>>> Training Basic BERT Baseline <<<")

    df_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")

    X_train_txt = df_train['text_clean'].tolist()
    y_train = df_train['sentiment'].values.astype(np.int32)
    X_val_txt = df_val['text_clean'].tolist()
    y_val = df_val['sentiment'].values.astype(np.int32)

    bert_cfg = cfg['bert']['basic']

    model = BERTClassifier(
        model_name=bert_cfg['model_name'],
        max_len=bert_cfg['max_len'],
        name='bert_basic_baseline'
    )

    model.train(
        X_train_txt, y_train, 
        epochs=bert_cfg['epochs'], 
        batch_size=bert_cfg['batch_size'],
        lr=float(bert_cfg['learning_rate']),
        patience=bert_cfg['patience']
    )

    model.evaluate(X_val_txt, y_val, name="val/bert")
    bert_dest = MODELS_DIR / "bert_basic"
    model.save(str(bert_dest))
    print(f"BERT Model saved to {bert_dest}")


if __name__ == "__main__":
    main()

# ===== File: scripts/train_specialist.py =====
#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.models.deep import BERTClassifier
from src.utils.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.features.builder import load_representation

logging.getLogger("transformers").setLevel(logging.ERROR)

def safe_binary_probs(probs):
    if probs.ndim == 2:
        return probs[:, 1]
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / args.best_params) as f:
        best = yaml.safe_load(f)

    cfg.setdefault('features', {})
    cfg.setdefault('models', {})
    cfg.setdefault('model', 'linear_svm')
    cfg.setdefault('cascade', {})

    cfg['features'].update(best.get('features', {}))
    cfg['model'] = best.get('model', cfg['model'])
    if 'models' in best:
        cfg['models'].update(best['models'])
    if 'cascade' in best:
        cfg['cascade'].update(best['cascade'])

    MODELS_DIR.mkdir(exist_ok=True)
    raw_preds_dir = RESULTS_DIR / "test" / "raw_predictions"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']
    token_col = "tokens_lower" 

    print("Loading data...")
    # Load custom representations (Raw Sparse Matrices)
    X_train_sp, y_train = load_representation(token_col, nc, w, z, 'train')
    X_val_sp, y_val = load_representation(token_col, nc, w, z, 'val')
    X_test_sp, y_test = load_representation(token_col, nc, w, z, 'test')

    # Load raw text for BERT
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")
    X_val_txt = df_val['text_clean'].tolist()
    X_test_txt = df_test['text_clean'].tolist()

    # Create STAGE-1 TF-IDF feature space (based on full Training data)
    tfidf_base = TfidfTransformer()
    X_train_tfidf = tfidf_base.fit_transform(X_train_sp)
    X_val_tfidf = tfidf_base.transform(X_val_sp)

    # ==========================================
    # 1. TRAIN STAGE-1 BASE MODEL
    # ==========================================
    print("\nTraining Stage-1 Base Model on full Training Set...")
    if cfg['model'] == 'linear_svm':
        base_model = LinearSVMClassifier(C=cfg['models']['linear_svm']['C'], name='Base_SVM')
    else:
        base_model = LogisticRegressionClassifier(C=cfg['models']['logreg']['C'], name='Base_LogReg')
    
    base_model.train(X_train_tfidf, y_train)

    # ==========================================
    # 2. MINE HARD SAMPLES FROM VALIDATION
    # ==========================================
    threshold = cfg['cascade']['delegation_threshold']
    lower = threshold
    upper = 1.0 - lower

    print("\nMining hard samples from the Validation set...")
    probs_val = safe_binary_probs(base_model.predict_proba(X_val_tfidf))
    mask_uncertain = (probs_val >= lower) & (probs_val <= upper)
    hard_indices = np.where(mask_uncertain)[0]
    
    print(f"Validation hard samples found: {len(hard_indices)} ({len(hard_indices)/len(y_val):.2%})")

    # BUILD SPECIALIST TF-IDF SPACE
    X_hard_sp = X_val_sp[hard_indices]

    # NEW TF-IDF tuned to the vocabulary of the hard cases
    print("Fitting new TF-IDF space exclusively on hard samples...")
    tfidf_spec = TfidfTransformer()
    X_hard_tfidf_spec = tfidf_spec.fit_transform(X_hard_sp)

    # Transform the test set using this new specialist TF-IDF space
    X_test_tfidf_spec = tfidf_spec.transform(X_test_sp)

    # Slice text and arrays for BERT
    X_hard_txt = [X_val_txt[i] for i in hard_indices]
    y_hard = np.array([y_val[i] for i in hard_indices], dtype=np.int32)

    # --- Specialist SVM ---
    print("\nTraining Specialist SVM on hard samples...")
    svm_spec = LinearSVMClassifier(name='Spec_SVM')
    svm_spec.train(X_hard_tfidf_spec, y_hard)

    probs_svm_spec = safe_binary_probs(svm_spec.predict_proba(X_test_tfidf_spec))
    pd.DataFrame({'true_label': y_test, 'probability': probs_svm_spec}).to_csv(
        raw_preds_dir / "svm_specialist.csv", index=False
    )
    print(f"-> Specialist SVM Test Accuracy: {( (probs_svm_spec > 0.5).astype(int) == y_test ).mean():.4f}")

    # --- Specialist Logistic Regression ---
    print("\nTraining Specialist LogReg on hard samples...")
    lr_spec = LogisticRegressionClassifier(name='Spec_LogReg')
    lr_spec.train(X_hard_tfidf_spec, y_hard)
    
    probs_lr_spec = safe_binary_probs(lr_spec.predict_proba(X_test_tfidf_spec))
    pd.DataFrame({'true_label': y_test, 'probability': probs_lr_spec}).to_csv(
        raw_preds_dir / "logreg_specialist.csv", index=False
    )
    print(f"-> Specialist LogReg Test Accuracy: {( (probs_lr_spec > 0.5).astype(int) == y_test ).mean():.4f}")

    # --- Specialist BERT ---
    print("\nFine-tuning Specialist BERT on hard samples...")
    bert_basic_path = MODELS_DIR / "bert_basic"
    if not bert_basic_path.exists():
        print("Skipping BERT: Basic BERT model not found.")
    else:
        bert_spec = BERTClassifier.load(str(bert_basic_path), name="BERT_Specialist")
        bert_spec.freeze_backbone(num_layers_to_freeze=4)

        spec_cfg = cfg['bert']['specialist']
        bert_spec.train(
            X_hard_txt, y_hard,
            epochs=int(spec_cfg['epochs']),
            batch_size=int(spec_cfg['batch_size']),
            lr=float(spec_cfg['learning_rate']),
            patience=int(spec_cfg['patience'])
        )

        probs_bert_spec = safe_binary_probs(bert_spec.predict_proba(X_test_txt))
        pd.DataFrame({'true_label': y_test, 'probability': probs_bert_spec}).to_csv(
            raw_preds_dir / "bert_specialist.csv", index=False
        )
        print(f"-> Specialist BERT Test Accuracy: {( (probs_bert_spec > 0.5).astype(int) == y_test ).mean():.4f}")

    print("\nAll specialists trained and test predictions saved!")

if __name__ == "__main__":
    main()

# ===== File: scripts/visualize_ablation_study.py =====
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

    # --- HIGHLIGHT THE BEST BAR ---
    for ax in g.axes.flat:
        best_width = 0
        best_patch = None

        # 1. Iterate through every bar drawn on this specific subplot
        for patch in ax.patches:
            # get_width() on a horizontal bar chart returns the actual F1-score value
            width = patch.get_width() 
            if width > best_width:
                best_width = width
                best_patch = patch

        # 2. Annotate the winning bar
        if best_patch:
            # Draw the F1-score text slightly to the right of the bar
            ax.text(
                best_width + 0.0005,  # X-position (push right slightly)
                best_patch.get_y() + best_patch.get_height() / 2,  # Y-position (center vertically)
                f"★ {best_width:.4f}", 
                color='#333333',
                fontsize=11,
                fontweight='bold',
                ha='left',
                va='center'
            )
            
            # Optional: Add a dark outline to the winning bar to make it pop visually
            best_patch.set_edgecolor('#333333')
            best_patch.set_linewidth(1.5)
    # ------------------------------
    
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

# ===== File: scripts/visualize_certainty.py =====
#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from src.utils.paths import RESULTS_DIR, FIGURES_DIR

# Define the colors globally to ensure consistency across subplots
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
    # UPDATED: Dynamically inject the experiment_name into the path
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

    # We will remove individual legends and create one unified legend for the whole figure
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    return True

def plot_model_comparison(svm_file, logreg_file, experiment_name):
    sns.set_theme(style="whitegrid", context="talk")

    # Create a 1x2 grid of subplots sharing the Y-axis
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # UPDATED: Pass experiment_name into the plotting functions
    process_and_plot_axis(axes[0], svm_file, "Linear SVM (1-3 N-Grams, Z=2.0)", experiment_name)
    process_and_plot_axis(axes[1], logreg_file, "Logistic Regression (1-3 N-Grams, Z=2.0)", experiment_name)

    # Only the leftmost plot needs the Y-axis label
    axes[0].set_ylabel("Number of Samples", fontsize=14, fontweight='bold')
    if len(axes) > 1:
        axes[1].set_ylabel("") # Clear just in case

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
    # Adjust layout to make room for the unified legend at the bottom
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

# ===== File: scripts/visualize_cluster_comparison.py =====
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

    # 1. Load Baselines
    baselines = {}
    for model_key, label in [("linear_svm", "Linear SVM"), ("logreg", "Logistic Regression")]:

        base_file = reports_path / f"{model_key}_k0_w0.csv"
        if base_file.exists():
            df = pd.read_csv(base_file, index_col=0)
            baselines[label] = df.loc["macro avg", "f1-score"]
        else:
            print(f"Warning: Could not find baseline file {base_file.name}")

    # 2. Load Grid Search Cluster Results
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

    # Add Baseline Markers & Unified Legend
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

    # 5. Styling & Export
    g.set_axis_labels("Macro F1-Score", "Number of Concepts (k)")
    g.despine(left=True)
    g.fig.suptitle("Grid Search: Clustered Concepts vs. Standard Processing", y=1.02, fontweight="bold")

    # Dynamically set X-axis limits so the differences are visible
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

# ===== File: scripts/visualize_concepts.py =====
"""
Visualize concepts (clusters) for two sentiment weights, e.g., 0 and 10.
Loads concept mapping and stats, displays top positive/negative clusters.
"""
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.utils.paths import DATA_DIR, FIGURES_DIR


def load_concept_data(token_col, n_concepts, sentiment_weight):
    """Load concept mapping and stats for a given configuration."""
    # Concept mapping (unit_to_cluster)
    concept_path = DATA_DIR / "concepts" / token_col / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not concept_path.exists():
        raise FileNotFoundError(f"Concept file not found: {concept_path}")
    with open(concept_path, "rb") as f:
        concept_data = pickle.load(f)
    unit_to_cluster = concept_data['unit_to_cluster']
    n_concepts_actual = concept_data['n_concepts']

    # Stats (logodds per class) – saved by builder.compute_concept_z_indices
    stats_path = DATA_DIR / "stats" / token_col / "concepts" / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)   # dict {0: df, 1: df}

    return unit_to_cluster, stats, n_concepts_actual


def plot_top_concepts(unit_to_cluster, stats, sentiment_weight, top_n=15, save_path=None):
    """Lollipop plot of top positive and negative concepts by Z-score."""
    pos_df = stats[1].sort_values('zscore', ascending=False).head(top_n)
    neg_df = stats[0].sort_values('zscore', ascending=False).head(top_n)

    # Representative word for each concept (first word in cluster)
    concept_repr = {}
    for word, cid in unit_to_cluster.items():
        if cid not in concept_repr:
            concept_repr[cid] = word.replace(' ', '_')

    pos_labels = [concept_repr.get(c, f"c{c}") for c in pos_df['concept']]
    neg_labels = [concept_repr.get(c, f"c{c}") for c in neg_df['concept']]

    plot_data = pd.DataFrame({
        'Concept': pos_labels + neg_labels,
        'Z-score': pos_df['zscore'].tolist() + [-x for x in neg_df['zscore'].tolist()],
        'Sentiment': ['Positive'] * top_n + ['Negative'] * top_n
    }).sort_values('Z-score')

    plt.figure(figsize=(10, 8))
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c'}
    for sent, color in colors.items():
        subset = plot_data[plot_data['Sentiment'] == sent]
        plt.hlines(y=subset['Concept'], xmin=0, xmax=subset['Z-score'],
                   color=color, alpha=0.5, linewidth=2)
        plt.scatter(subset['Z-score'], subset['Concept'],
                    color=color, s=80, label=sent, edgecolors='white', zorder=3)

    plt.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    plt.title(f"Top {top_n} Discriminative Concepts (weight = {sentiment_weight})", fontsize=14)
    plt.xlabel("Sentiment Strength (Z-score)")
    plt.ylabel("Concept Representative")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved lollipop plot to {save_path}")


def plot_cluster_wordclouds(unit_to_cluster, stats, sentiment_weight, class_id=1, top_n=5, max_words=50, save_path=None):
    """Word clouds for top N clusters of a given class (positive=1, negative=0)."""
    # Group words by cluster
    cluster_to_words = {}
    for word, cid in unit_to_cluster.items():
        cluster_to_words.setdefault(cid, []).append(word.replace(' ', '_'))

    class_df = stats[class_id].sort_values('zscore', ascending=False).head(top_n)
    top_clusters = class_df['concept'].tolist()

    fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 5))
    if top_n == 1:
        axes = [axes]
    label = "Positive" if class_id == 1 else "Negative"
    fig.suptitle(f"Top {top_n} {label} Clusters (weight={sentiment_weight})", fontsize=16)

    for i, cid in enumerate(top_clusters):
        words = cluster_to_words.get(cid, ["empty"])
        freq = {w: 1 for w in words}   # equal weight; could use counts if available
        wc = WordCloud(width=400, height=400, background_color='white',
                       colormap='Greens' if class_id == 1 else 'Reds',
                       max_words=max_words).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"Cluster {cid}\nZ={class_df.iloc[i]['zscore']:.2f}", fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved word cloud grid to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_col', default='tokens_lower', help='Token column name')
    parser.add_argument('--n_concepts', type=int, default=5000, help='Number of concepts')
    parser.add_argument('--weights', nargs=2, type=float, default=[0.0, 10.0],
                        help='Two sentiment weights to compare')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top concepts to display')
    args = parser.parse_args()

    for w in args.weights:
        print(f"\n=== Processing weight = {w} ===")
        unit_to_cluster, stats, n_actual = load_concept_data(args.token_col, args.n_concepts, w)
        print(f"Loaded {len(unit_to_cluster)} units mapped to {n_actual} concepts.")

        # Lollipop plot
        lollipop_path = FIGURES_DIR / "concepts" / f"lollipop_{args.token_col}_k{args.n_concepts}_w{int(w)}.png"
        lollipop_path.parent.mkdir(parents=True, exist_ok=True)
        plot_top_concepts(unit_to_cluster, stats, w, top_n=args.top_n, save_path=lollipop_path)

        # Word clouds for positive
        pos_wc_path = FIGURES_DIR / "concepts" / f"pos_wc_{args.token_col}_k{args.n_concepts}_w{int(w)}.png"
        plot_cluster_wordclouds(unit_to_cluster, stats, w, class_id=1, top_n=5,
                                save_path=pos_wc_path)

        # Word clouds for negative
        neg_wc_path = FIGURES_DIR / "concepts" / f"neg_wc_{args.token_col}_k{args.n_concepts}_w{int(w)}.png"
        plot_cluster_wordclouds(unit_to_cluster, stats, w, class_id=0, top_n=5,
                                save_path=neg_wc_path)


if __name__ == "__main__":
    main()

# ===== File: scripts/visualize_custom_filtering.py =====
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.features.sentiment import SentimentFeatures
from src.utils.paths import FIGURES_DIR
from src.features.vectorizer import build_count_matrix
from src.features.selection import compute_global_mask, compute_class_mask


def main():
    # --- SETUP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Generate WordClouds for specific token columns.")
    parser.add_argument(
        "--column", 
        type=str, 
        default="tokens_lower", 
        help="The dataframe column containing the tokens to visualize (e.g., tokens_lower, tokens_filtered)"
    )
    args = parser.parse_args()
    target_col = args.column

    print("Loading data for visualization...")
    train_df = pd.read_parquet("data/preprocessed/train.parquet")
    
    if target_col not in train_df.columns:
        raise ValueError(f"Column '{target_col}' not found in the dataset. Available columns: {list(train_df.columns)}")
        
    y_train = train_df["sentiment"].values

    print(f"Vectorizing text for '{target_col}' (max_df=0.7)...")
    X_train, _, count_vect = build_count_matrix(train_df[target_col], None, max_df=0.7)
    vocab = count_vect.get_feature_names_out()

    print("Calculating Z-scores on the Custom matrix...")
    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    df_scores = sf.logodds_per_class[1].copy()
    df_scores['word'] = [vocab[i] for i in df_scores['concept']]

    z_threshold = 2

    # --- SPLIT INTO TRASHED AND KEPT ---
    trashed_df = df_scores[df_scores['zscore'].abs() <= z_threshold].copy()
    trashed_df['abs_z'] = trashed_df['zscore'].abs()
    trashed_df = trashed_df.sort_values('abs_z', ascending=True)

    kept_pos = df_scores[df_scores['zscore'] > z_threshold].sort_values('zscore', ascending=False)
    kept_neg = df_scores[df_scores['zscore'] < -z_threshold].sort_values('zscore', ascending=True)

    # --- GENERATE VISUALIZATIONS ---
    print("\nGenerating WordClouds (with punctuation preserved)...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Sentiment Analysis: {target_col}", fontsize=20, fontweight='bold', y=1.05)

    # Panel 1: Trashed (Greys) - using \S+ to keep punctuation
    trashed_text = " ".join([w.replace(" ", "_") for w in trashed_df['word'].head(150)])
    wc_trash = WordCloud(background_color='white', colormap='Greys', width=400, height=400, regexp=r"\S+").generate(trashed_text)
    axes[0].imshow(wc_trash, interpolation='bilinear')
    axes[0].set_title(f"Trashed Words (Neutral Noise)\nTotal Discarded: {len(trashed_df)}", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Kept Positive (Greens)
    pos_text = " ".join([w.replace(" ", "_") for w in kept_pos['word'].head(150)])
    wc_pos = WordCloud(background_color='white', colormap='Greens', width=400, height=400, regexp=r"\S+").generate(pos_text)
    axes[1].imshow(wc_pos, interpolation='bilinear')
    axes[1].set_title(f"Kept Positive Words\nTotal Kept: {len(kept_pos)}", fontsize=16, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Kept Negative (Reds)
    neg_text = " ".join([w.replace(" ", "_") for w in kept_neg['word'].head(150)])
    wc_neg = WordCloud(background_color='white', colormap='Reds', width=400, height=400, regexp=r"\S+").generate(neg_text)
    axes[2].imshow(wc_neg, interpolation='bilinear')
    axes[2].set_title(f"Kept Negative Words\nTotal Kept: {len(kept_neg)}", fontsize=16, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    
    # Update the save path to include the target column name dynamically
    out_path = FIGURES_DIR / "analysis" / f"{target_col}_wordclouds.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison image to {out_path}")


if __name__ == "__main__":
    main()

# ===== File: scripts/visualize_sentiment.py =====
#!/usr/bin/env python3
"""
Visualizes the specific words or concepts that drive sentiment.
Includes individual Top 5 Positive and Top 5 Negative Cluster visualization.
"""
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from src.utils.paths import get_figure_path, DATA_DIR

sns.set_theme(style="whitegrid")

def load_artifacts(ngram_range, n_concepts, weight):
    """Loads vocabulary, concepts, and sentiment statistics."""
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    
    # 1. Load Vocabulary
    vocab_path = DATA_DIR / "vocab" / f"vocab_{key}.pkl"
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found: {vocab_path}")
        return None, None, None

    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # 2. Load Concept Mapping (if applicable)
    concept_data = None
    if n_concepts > 0:
        filename = f"concepts_{key}_k{n_concepts}_w{int(weight)}.pkl"
        concept_path = DATA_DIR / "concepts" / filename
        
        if not concept_path.exists():
            print(f"❌ Concept file not found: {concept_path}")
            return None, None, None

        with open(concept_path, 'rb') as f:
            concept_data = pickle.load(f)
        
        # Generate feature names for concepts (since we don't have 'concept_units')
        feature_names = [f"Concept_{i}" for i in range(n_concepts)]
        repr_key = f"{key}_k{n_concepts}_w{int(weight)}"
    else:
        feature_names = vocab_data['vocab']
        repr_key = f"{key}_raw"

    # 3. Load Statistics (Z-Scores)
    stats_path = DATA_DIR / "stats" / f"stats_{repr_key}.pkl"
    if not stats_path.exists():
        print(f"❌ Stats not found: {stats_path}")
        return None, None, None

    with open(stats_path, 'rb') as f:
        logodds = pickle.load(f)
        
    return feature_names, logodds, concept_data

def plot_sentiment_importance(feature_names, logodds, title, output_path):
    """Generates the lollipop chart for top discriminative features."""
    top_pos = logodds[1].sort_values("zscore", ascending=False).head(15)
    top_neg = logodds[0].sort_values("zscore", ascending=False).head(15)
    
    pos_words = [feature_names[i] for i in top_pos['concept']]
    neg_words = [feature_names[i] for i in top_neg['concept']]
    
    df_plot = pd.DataFrame({
        "Feature": pos_words + neg_words,
        "Z-Score": list(top_pos['zscore']) + list(-top_neg['zscore']), 
        "Sentiment": ["Positive"] * 15 + ["Negative"] * 15
    }).sort_values("Z-Score")
    
    plt.figure(figsize=(10, 8))
    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c"}
    
    plt.hlines(y=df_plot["Feature"], xmin=0, xmax=df_plot["Z-Score"], 
               color=[colors[s] for s in df_plot["Sentiment"]], alpha=0.5, linewidth=2)
    
    for sentiment, color in colors.items():
        mask = df_plot["Sentiment"] == sentiment
        plt.scatter(df_plot.loc[mask, "Z-Score"], df_plot.loc[mask, "Feature"], 
                    color=color, s=80, label=sentiment, edgecolors='white', zorder=3)
        
    plt.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel("Sentiment Strength (Z-Score)")
    plt.ylabel("Feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved importance plot to {output_path}")

def plot_top_cluster_grid(concept_data, logodds, class_id, label, weight, top_n=5, n_concepts=None):
    """Generates a grid of word clouds for the top N clusters of a specific sentiment."""
    if not concept_data:
        return

    unit_to_cluster = concept_data['unit_to_cluster']
    cluster_contents = {}
    for word, cid in unit_to_cluster.items():
        cluster_contents.setdefault(cid, []).append(word.replace(" ", "_"))

    top_clusters = logodds[class_id].sort_values("zscore", ascending=False).head(top_n)
    
    fig, axes = plt.subplots(1, top_n, figsize=(22, 5))
    fig.suptitle(f"Top {top_n} {label} Semantic Clusters (Weight: {weight})", fontsize=18)

    for i, (_, row) in enumerate(top_clusters.iterrows()):
        cid = int(row['concept'])
        zscore = row['zscore']
        words = cluster_contents.get(cid, ["empty"])
        
        from collections import Counter
        word_freq = Counter(words)
        
        wc = WordCloud(width=400, height=400, background_color="white", 
                       colormap="Greens" if class_id == 1 else "Reds").generate_from_frequencies(word_freq)
        
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"Cluster {cid}\nZ: {zscore:.2f}", fontsize=12)
        axes[i].axis("off")

    out_path = get_figure_path("vocabulary", f"top_{top_n}_{label}_w{int(weight)}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved {label} cluster grid to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmin", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=3)
    parser.add_argument("--n_concepts", type=int, default=5000)
    parser.add_argument("--sentiment_weight", type=float, default=0.0)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    
    print(f"--- Visualizing: Concepts={args.n_concepts}, Weight={args.sentiment_weight} ---")
    
    feature_names, logodds, concept_data = load_artifacts(
        (args.nmin, args.nmax), 
        args.n_concepts, 
        args.sentiment_weight
    )
    
    if feature_names is None:
        return

    run_name = f"ngram_{args.nmin}_{args.nmax}_k{args.n_concepts}_w{int(args.sentiment_weight)}"
    title = f"Weight: {args.sentiment_weight} | K: {args.n_concepts}"
    
    # 1. Lollipop Importance Plot
    importance_path = get_figure_path("vocabulary", f"{run_name}_importance.png")
    plot_sentiment_importance(feature_names, logodds, title, importance_path)

    # 2. Individual Cluster Grids (if using concepts)
    if args.n_concepts > 0:
        plot_top_cluster_grid(concept_data, logodds, 1, "Positive", args.sentiment_weight, args.top_n)
        plot_top_cluster_grid(concept_data, logodds, 0, "Negative", args.sentiment_weight, args.top_n)

if __name__ == "__main__":
    main()

# ===== File: scripts/visualize_thesis.py =====
#!/usr/bin/env python3
"""
Generate the final thesis figure comparing all systems.
- Loads ensemble_results.csv (produced by run_ensemble.py)
- Creates a bar chart of accuracies and saves it.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.paths import RESULTS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid", context="talk")

def main():
    results_path = RESULTS_DIR / "thesis" / "ensemble_results.csv"
    if not results_path.exists():
        raise FileNotFoundError("ensemble_results.csv not found. Run scripts/run_ensemble.py first.")

    df = pd.read_csv(results_path)
    # Extract accuracy columns (all except Delegation Rate and Threshold)
    acc_cols = [c for c in df.columns if "Rate" not in c and "Threshold" not in c]
    acc_df = df[acc_cols].melt(var_name="System", value_name="Accuracy")

    # Sort systems by accuracy
    order = acc_df.groupby("System")["Accuracy"].max().sort_values(ascending=False).index

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=acc_df, x="System", y="Accuracy", hue="System", order=order, palette="viridis", legend=False)
    ax.set_ylim(0.8, 1.0)  # adjust as needed
    ax.set_title("Final Thesis: Cascade Performance Comparison", fontsize=16, fontweight='bold')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha='right')

    # Add value labels on bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=11, color='black')

    plt.tight_layout()
    out_path = FIGURES_DIR / "thesis" / "thesis_final_results.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Thesis figure saved to {out_path}")

    # Also save a CSV with the accuracy comparison
    acc_df.to_csv(FIGURES_DIR / "thesis" / "thesis_accuracy_table.csv", index=False)

if __name__ == "__main__":
    main()

# ===== File: src/features/builder.py =====
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
import torch
from sentence_transformers import SentenceTransformer
import faiss
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.paths import DATA_DIR
from src.features.vectorizer import build_count_matrix
from src.features.sentiment import SentimentFeatures

_SENTENCE_MODEL = None
_ANALYZER = None


def _get_sentence_model(model_name: str = 'all-MiniLM-L6-v2'):
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _SENTENCE_MODEL = SentenceTransformer(model_name).to(device)
    return _SENTENCE_MODEL


def _get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def build_unit_matrices(token_col, ngram_range=(1,3), min_df=10, max_df=0.7, force=False):
    cache_dir = DATA_DIR / "cache_matrices" / token_col
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train.npz"
    val_path = cache_dir / "val.npz"
    test_path = cache_dir / "test.npz"
    vocab_path = cache_dir / "vocab.pkl"

    print(f"Building unit matrices for '{token_col}' ...")
    train_df = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")

    # Prepare token lists (each is a list of tokens)
    train_tokens = train_df[token_col].tolist()
    val_tokens = val_df[token_col].tolist()
    test_tokens = test_df[token_col].tolist()

    # Build matrices for train and val simultaneously
    X_train, X_val, vectorizer = build_count_matrix(
        train_tokens, val_tokens,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range
    )

    # Build test matrix using the same vectorizer
    test_reviews = [" ".join(tokens) for tokens in test_tokens]
    X_test = vectorizer.transform(test_reviews)

    # Save matrices
    save_npz(train_path, X_train)
    save_npz(val_path, X_val)
    save_npz(test_path, X_test)

    # Save vocabulary (feature names)
    with open(vocab_path, "wb") as f:
        pickle.dump(vectorizer.get_feature_names_out(), f)

    print(f"Unit matrices for '{token_col}' saved.")
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'vocab': vocab_path,
    }


def compute_unit_z_indices(token_col, z_scores, force=False):
    cache_dir = DATA_DIR / "cache_matrices" / token_col / "unit_z_indices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load train matrix
    train_path = DATA_DIR / "cache_matrices" / token_col / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Train matrix for '{token_col}' not found.")
    X_train = load_npz(train_path)
    y_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")['sentiment'].values

    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    for z in z_scores:
        keep_set = sf.filter_by_zscore(z)
        keep_indices = sorted(keep_set)
        out_path = cache_dir / f"z_{z}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(keep_indices, f)
        print(f"   Unit Z={z}: kept {len(keep_indices)} features.")


def compute_embeddings(token_col, ngram_range=(1,3), model_name='all-MiniLM-L6-v2'):
    cache_dir = DATA_DIR / "cache_matrices" / token_col
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npz"
    if emb_path.exists():
        return

    vocab_path = cache_dir / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary for '{token_col}' not found.")
    with open(vocab_path, "rb") as f:
        units = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name).to(device)
    embeddings = model.encode(units, batch_size=256, convert_to_tensor=True,
                              device=device, show_progress_bar=False)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = np.array([analyzer.polarity_scores(u)['compound'] for u in units],
                                dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(emb_path, embeddings=embeddings, sentiment_scores=sentiment_scores)
    print(f"Embeddings for '{token_col}' saved.")


def extract_concepts(token_col, n_concepts, sentiment_weight, force=False):
    out_dir = DATA_DIR / "concepts" / token_col
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"

    # Load units and embeddings
    vocab_path = DATA_DIR / "cache_matrices" / token_col / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary for '{token_col}' not found.")
    with open(vocab_path, "rb") as f:
        units = pickle.load(f)

    emb_path = DATA_DIR / "cache_matrices" / token_col / "embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings for '{token_col}' not found.")
    data = np.load(emb_path)

    # Augment embeddings with weighted sentiment
    aug_embeddings = np.hstack([
        data['embeddings'],
        data['sentiment_scores'] * float(sentiment_weight)
    ]).astype(np.float32)
    aug_embeddings = aug_embeddings / np.linalg.norm(aug_embeddings, axis=1, keepdims=True)

    actual_k = min(n_concepts, len(units))
    kmeans = faiss.Kmeans(aug_embeddings.shape[1], actual_k, niter=20,
                          verbose=False, gpu=torch.cuda.is_available())
    kmeans.train(aug_embeddings)
    _, labels = kmeans.index.search(aug_embeddings, 1)

    centroids = kmeans.centroids
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    with open(out_path, "wb") as f:
        pickle.dump({
            'unit_to_cluster': dict(zip(units, labels.flatten().tolist())),
            'centroids': centroids,
            'sentiment_weight': sentiment_weight,
            'n_concepts': actual_k
        }, f)
    print(f"Concepts for '{token_col}' k={n_concepts} w={sentiment_weight} saved.")


def build_concept_matrices(token_col, n_concepts, sentiment_weight, force=False):
    concept_dir = DATA_DIR / "cache_matrices" / token_col / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    concept_key = f"k{n_concepts}_w{int(sentiment_weight)}"
    out_train = concept_dir / f"{concept_key}_train.npz"
    out_val = concept_dir / f"{concept_key}_val.npz"
    out_test = concept_dir / f"{concept_key}_test.npz"

    # Load unit‑to‑concept mapping
    map_path = DATA_DIR / "concepts" / token_col / f"k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not map_path.exists():
        raise FileNotFoundError(f"Concept mapping for '{token_col}' {concept_key} not found.")
    with open(map_path, "rb") as f:
        map_data = pickle.load(f)
    unit_to_cluster = map_data['unit_to_cluster']
    n_concepts_actual = map_data['n_concepts']

    # Build mapping from unit index to concept index
    # Load vocabulary to get unit index
    vocab_path = DATA_DIR / "cache_matrices" / token_col / "vocab.pkl"
    with open(vocab_path, "rb") as f:
        units = pickle.load(f)
    unit_to_idx = {u: i for i, u in enumerate(units)}
    idx_to_cluster = {}
    for u, cid in unit_to_cluster.items():
        idx = unit_to_idx.get(u)
        if idx is not None:
            idx_to_cluster[idx] = cid

    # Helper to remap a unit matrix to concept matrix
    def remap_unit_matrix(unit_mat_path, out_path, n_docs):
        unit_mat = load_npz(unit_mat_path)
        # Convert to COO for easy manipulation
        coo = unit_mat.tocoo()
        rows = coo.row
        cols = coo.col
        data = coo.data
        # Map unit columns to concept columns
        new_cols = np.array([idx_to_cluster.get(c, -1) for c in cols], dtype=np.int32)
        mask = new_cols != -1
        rows = rows[mask]
        new_cols = new_cols[mask]
        data = data[mask]
        if len(rows) == 0:
            concept_mat = csr_matrix((n_docs, 0), dtype=np.float32)
        else:
            concept_mat = coo_matrix((data, (rows, new_cols)),
                                     shape=(n_docs, n_concepts_actual)).tocsr()
        save_npz(out_path, concept_mat)

    # Get number of documents per split
    n_train = len(pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet"))
    n_val   = len(pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet"))
    n_test  = len(pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet"))

    unit_train = DATA_DIR / "cache_matrices" / token_col / "train.npz"
    unit_val   = DATA_DIR / "cache_matrices" / token_col / "val.npz"
    unit_test  = DATA_DIR / "cache_matrices" / token_col / "test.npz"

    remap_unit_matrix(unit_train, out_train, n_train)
    remap_unit_matrix(unit_val,   out_val,   n_val)
    remap_unit_matrix(unit_test,  out_test,  n_test)

    print(f"Concept matrices for '{token_col}' {concept_key} saved.")


def compute_concept_z_indices(token_col, n_concepts, sentiment_weight, z_scores, force=False):
    concept_dir = DATA_DIR / "cache_matrices" / token_col / "concepts"
    concept_key = f"k{n_concepts}_w{int(sentiment_weight)}"
    indices_dir = concept_dir / "z_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    stats_dir = DATA_DIR / "stats" / token_col / "concepts"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{concept_key}.pkl"

    # Check if everything already exists
    indices_exist = all((indices_dir / f"{concept_key}_z{z}.pkl").exists() for z in z_scores)
    stats_exist = stats_path.exists()
    if not force and indices_exist and stats_exist:
        print(f"✅ Concept Z‑score indices and stats for '{token_col}' {concept_key} already exist.")
        return

    # Load concept train matrix
    train_path = concept_dir / f"{concept_key}_train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Concept train matrix for '{token_col}' {concept_key} not found.")
    X_train = load_npz(train_path)
    y_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")['sentiment'].values

    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())

    # Save keep indices for each z
    for z in z_scores:
        keep_set = sf.filter_by_zscore(z)
        keep_indices = sorted(keep_set)
        out_path = indices_dir / f"{concept_key}_z{z}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(keep_indices, f)
        print(f"   Concept Z={z}: kept {len(keep_indices)} features.")

    # Save full stats
    with open(stats_path, "wb") as f:
        pickle.dump(sf.logodds_per_class, f)
    print(f"   Saved full stats to {stats_path}")


def load_representation(token_col, n_concepts, sentiment_weight, z_threshold, split):
    y = pd.read_parquet(DATA_DIR / "preprocessed" / f"{split}.parquet")['sentiment'].values

    if n_concepts == 0:
        # Unit level
        unit_path = DATA_DIR / "cache_matrices" / token_col / f"{split}.npz"
        if not unit_path.exists():
            raise FileNotFoundError(f"Unit matrix for '{token_col}' split {split} not found.")
        X = load_npz(unit_path)

        # Load unit Z‑score indices
        indices_dir = DATA_DIR / "cache_matrices" / token_col / "unit_z_indices"
        indices_path = indices_dir / f"z_{z_threshold}.pkl"
        if not indices_path.exists():
            raise FileNotFoundError(f"Unit Z‑score indices for z={z_threshold} not found.")
        with open(indices_path, "rb") as f:
            keep_indices = pickle.load(f)
        X = X[:, keep_indices]

    else:
        # Concept level
        concept_dir = DATA_DIR / "cache_matrices" / token_col / "concepts"
        concept_key = f"k{n_concepts}_w{int(sentiment_weight)}"
        mat_path = concept_dir / f"{concept_key}_{split}.npz"
        if not mat_path.exists():
            raise FileNotFoundError(f"Concept matrix for '{token_col}' {concept_key} split {split} not found.")
        X = load_npz(mat_path)

        # Load concept Z‑score indices
        indices_dir = concept_dir / "z_indices"
        indices_path = indices_dir / f"{concept_key}_z{z_threshold}.pkl"
        if not indices_path.exists():
            raise FileNotFoundError(f"Concept Z‑score indices for {concept_key} z={z_threshold} not found.")
        with open(indices_path, "rb") as f:
            keep_indices = pickle.load(f)
        X = X[:, keep_indices]

    return X, y

# ===== File: src/features/concept_remap.py =====
# src/features/concept_remap.py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def remap_sparse_matrix(X_unit: csr_matrix, unit_to_concept: dict, n_concepts: int = None):
    """
    Transform CSR matrix with columns = unit IDs into CSR matrix with columns = concept IDs.

    Parameters
    ----------
    X_unit : csr_matrix, shape (n_docs, n_units)
    unit_to_concept : dict
        Mapping from unit ID (int) to concept ID (int). Units not in mapping are dropped.
    n_concepts : int, optional
        Total number of concepts (must be >= max(concept ID)+1). If None, inferred.

    Returns
    -------
    X_concept : csr_matrix
    """
    X_unit = X_unit.tocoo()
    rows = X_unit.row
    cols = X_unit.col
    data = X_unit.data

    # Map column indices
    new_cols = np.array([unit_to_concept.get(c, -1) for c in cols], dtype=np.int32)
    mask = new_cols != -1
    rows = rows[mask]
    new_cols = new_cols[mask]
    data = data[mask]

    if len(rows) == 0:
        # No concepts mapped – return empty matrix
        return csr_matrix((X_unit.shape[0], 0), dtype=data.dtype)

    if n_concepts is None:
        n_concepts = max(new_cols) + 1

    X_concept = coo_matrix((data, (rows, new_cols)), shape=(X_unit.shape[0], n_concepts))
    return X_concept.tocsr()

# ===== File: src/features/concepts.py =====
# src/features/concepts.py
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional, Union
from sklearn.feature_selection import SelectKBest, f_regression


class ConceptExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        self.selected_dims = None   # for supervised dimension reduction
        self._embeddings_cache = None   # optional: store precomputed embeddings

    def set_precomputed_embeddings(self, embeddings: np.ndarray, units: List[str]):
        """
        Inject precomputed embeddings (e.g., from cache) to avoid recomputation.
        Embeddings must be normalized and in the same order as `units`.
        """
        self._embeddings_cache = (units, embeddings)

    def train_concepts(self,
                       units: List[str],
                       sentiment_map: Optional[Dict[str, float]] = None,
                       retention_percentile: int = 10,
                       n_clusters: int = 5000,
                       batch_size: int = 128,
                       printing: bool = True) -> Dict[str, Any]:
        """
        Cluster units into concepts.
        If embeddings have been precomputed (via set_precomputed_embeddings), they are used.
        """
        unique_units = sorted(list(dict.fromkeys(units)))
        n_clusters = min(n_clusters, len(unique_units))

        # --- Embeddings ---
        if self._embeddings_cache is not None:
            cached_units, cached_emb = self._embeddings_cache
            if cached_units == unique_units:
                if printing:
                    print("Using precomputed embeddings from cache.")
                embeddings_np = cached_emb
            else:
                raise ValueError("Cached embeddings do not match provided units.")
        else:
            if printing:
                print(f"Generating embeddings for {len(unique_units)} unique units...")
            with torch.inference_mode():
                embeddings = self.model.encode(
                    unique_units,
                    batch_size=batch_size,
                    show_progress_bar=printing,
                    convert_to_tensor=True,
                    device=self.device
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings_np = embeddings.cpu().float().numpy().astype(np.float32)
                del embeddings
                torch.cuda.empty_cache()

        # --- Supervised Dimension Reduction ---
        if sentiment_map is not None:
            if printing:
                print("Performing Supervised Dimension Reduction (Sentiment Focus)...")
            y = np.array([sentiment_map.get(u, 0.5) for u in unique_units], dtype=np.float32)

            k = int(embeddings_np.shape[1] * (retention_percentile / 100))
            k = max(k, 1)

            selector = SelectKBest(f_regression, k=k)
            selector.fit(embeddings_np, y)

            self.selected_dims = selector.get_support(indices=True)
            embeddings_np = embeddings_np[:, self.selected_dims]

            if printing:
                print(f"Reduced embedding dimensions from {selector.n_features_in_} to {k}.")

        # --- FAISS Clustering ---
        if printing:
            print(f"FAISS Clustering (n={n_clusters})...")
        d = embeddings_np.shape[1]

        # Use GPU if available
        gpu_res = None
        if faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True

        kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=printing, gpu=gpu_res is not None)
        kmeans.train(embeddings_np)
        cluster_centers = kmeans.centroids

        # Assign each unit to nearest centroid
        index = faiss.IndexFlatL2(d)
        if gpu_res:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(embeddings_np)
        _, labels = index.search(cluster_centers, 1)   # find closest unit to each centroid
        rep_indices = labels.flatten()
        concept_units = [unique_units[idx] for idx in rep_indices]

        # Actually assign units to clusters (using centroids)
        index.reset()
        index.add(cluster_centers)
        _, unit_labels = index.search(embeddings_np, 1)
        unit_labels = unit_labels.flatten()

        return {
            "cluster_centers": torch.tensor(cluster_centers),
            "concept_units": concept_units,
            "unit_to_cluster": {u: int(l) for u, l in zip(unique_units, unit_labels)},
            "n_concepts": n_clusters
        }

    def map_units_to_clusters(self,
                              units: List[str],
                              cluster_centers: torch.Tensor,
                              batch_size: int = 128,
                              printing: bool = True) -> Dict[str, int]:
        """
        Map new units to existing clusters (centroids).
        Uses cosine similarity (via L2 on normalized vectors) with threshold.
        """
        if not units:
            return {}
        unique_units = sorted(list(dict.fromkeys(units)))

        with torch.inference_mode():
            embeddings = self.model.encode(
                unique_units,
                batch_size=batch_size,
                show_progress_bar=printing,
                convert_to_tensor=True,
                device=self.device
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().float().numpy()
            del embeddings
            torch.cuda.empty_cache()

        if self.selected_dims is not None:
            embeddings_np = embeddings_np[:, self.selected_dims]

        centers_np = cluster_centers.cpu().numpy().astype(np.float32)
        d = centers_np.shape[1]

        # Build index on centroids
        index = faiss.IndexFlatL2(d)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(centers_np)

        distances, labels = index.search(embeddings_np, 1)
        # Convert L2^2 to cosine similarity (since vectors are normalized)
        similarities = 1 - distances.flatten() / 2
        SIM_THRESHOLD = 0.6

        result = {}
        for u, l, s in zip(unique_units, labels.flatten(), similarities):
            if s >= SIM_THRESHOLD:
                result[u] = int(l)
        return result

# ===== File: src/features/selection.py =====
import numpy as np


def compute_global_mask(X_train, max_df_ratio=0.7):
    """Return indices of features with document frequency <= max_df_ratio."""
    doc_freqs = np.array((X_train > 0).sum(axis=0)).flatten()
    max_count = max_df_ratio * X_train.shape[0]
    return np.where(doc_freqs <= max_count)[0]


def compute_class_mask(X_train, y_train, max_df_ratio=0.7):
    """
    Return indices of features that satisfy class‑specific max_df:
        df_pos <= max_df_ratio * n_pos  OR  df_neg <= max_df_ratio * n_neg
    """
    mask_pos = (y_train == 1)
    mask_neg = (y_train == 0)
    df_pos = np.array((X_train[mask_pos] > 0).sum(axis=0)).flatten()
    df_neg = np.array((X_train[mask_neg] > 0).sum(axis=0)).flatten()
    max_pos = max_df_ratio * mask_pos.sum()
    max_neg = max_df_ratio * mask_neg.sum()
    return np.where((df_pos <= max_pos) | (df_neg <= max_neg))[0]

# ===== File: src/features/sentiment.py =====
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List


class SentimentFeatures:
    """
    Compute log-odds and Z-scores for each feature (concept or unit)
    using a pre-computed sparse CSR matrix.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.logodds_per_class = {}   # dict {0: df, 1: df}
        self.concept_list = None      # list of feature IDs (columns)
        self._feature_to_idx = None

    def fit(self, X: csr_matrix, y: List[int]):
        n_docs, n_features = X.shape
        self.concept_list = list(range(n_features))   # column indices are IDs
        self._feature_to_idx = {f: i for i, f in enumerate(self.concept_list)}

        y_arr = np.array(y)
        for cls in [0, 1]:
            mask = (y_arr == cls)
            counts_cls = X[mask].sum(axis=0).A1
            counts_not = X[~mask].sum(axis=0).A1

            p_cls = (counts_cls + self.alpha) / (counts_cls.sum() + self.alpha * n_features)
            p_not = (counts_not + self.alpha) / (counts_not.sum() + self.alpha * n_features)

            logodds = np.log(p_cls / p_not)
            z_scores = logodds / np.sqrt(1/(counts_cls + self.alpha) + 1/(counts_not + self.alpha))

            self.logodds_per_class[cls] = pd.DataFrame({
                "concept": self.concept_list,
                "zscore": z_scores
            })

        return self

    def filter_by_zscore(self, threshold: float) -> set:
        """
        Return set of concept IDs whose Z-score > threshold for either class.
        """
        pos_set = set(self.logodds_per_class[1][self.logodds_per_class[1]['zscore'] > threshold]['concept'])
        neg_set = set(self.logodds_per_class[0][self.logodds_per_class[0]['zscore'] > threshold]['concept'])
        return pos_set | neg_set

# ===== File: src/features/vectorizer.py =====
from sklearn.feature_extraction.text import CountVectorizer


def space_tokenizer(text):
    """Split text exactly by spaces (bypass sklearn's regex)."""
    return text.split(' ')


def build_count_matrix(train_tokens, val_tokens, min_df=10, max_df=1.0, ngram_range=(1,3)):

    train_reviews = [" ".join(tokens) for tokens in train_tokens]
    val_reviews = None if val_tokens is None else [" ".join(tokens) for tokens in val_tokens]

    vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=space_tokenizer,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    X_train = vectorizer.fit_transform(train_reviews)
    X_val = None if val_tokens is None else vectorizer.transform(val_reviews)
    return X_train, X_val, vectorizer

# ===== File: src/models/base_model.py =====
import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report
from src.utils.paths import RESULTS_DIR

class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def save(self, path: str):
        """Standardized save for Scikit-Learn based models."""
        if not path.endswith('.joblib'):
            path += '.joblib'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """Standardized load for Scikit-Learn based models."""
        if not path.endswith('.joblib'):
            path += '.joblib'
        return joblib.load(path)

    def train(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict_label(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def get_evaluation_data(self, y_true, probs) -> pd.DataFrame:
        return pd.DataFrame({
            'true_label': y_true,
            'probability': probs
        })

    def evaluate(self, X_test, y_test, name: str = "val"):
        probs = self.predict_proba(X_test)
        preds = self.predict_label(X_test)

        # --- PATHS ---
        data_dir = RESULTS_DIR / name / "raw_predictions"
        report_dir = RESULTS_DIR / name / "classification_reports"

        for d in [data_dir, report_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Save Raw Evaluation Data (Source for all future plots)
        eval_df = self.get_evaluation_data(y_test, probs)
        eval_df.to_csv(data_dir / f"{self.name}.csv", index=False)

        # 2. Save Standard Metrics
        report_dict = classification_report(y_test, preds, output_dict=True, digits=4)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(report_dir / f"{self.name}.csv")

# ===== File: src/models/classic.py =====
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.models.base_model import BaseModel
from sklearn.svm import SVC


class LinearSVMClassifier(BaseModel):
    """LinearSVC with probability calibration."""

    def __init__(self, C=1.0, name="linear_svm"):
        super().__init__(name)
        self.base_model = LinearSVC(C=C, max_iter=10000, random_state=42)
        self.model = CalibratedClassifierCV(self.base_model)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Returning probability for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]


class RbfSVMClassifier(BaseModel):
    """SVC with RBF kernel wrapped with probability calibration."""

    def __init__(self, C=1.0, gamma='scale', name="rbf_svm"):
        super().__init__(name)
        self.svm = SVC(C=C, kernel='rbf', gamma=gamma, probability=False, random_state=42, verbose=True)
        self.model = CalibratedClassifierCV(self.svm)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Returning probability for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]


class LogisticRegressionClassifier(BaseModel):
    """Standardowa Regresja Logistyczna."""

    def __init__(self, C=1.0, name="logreg"):
        super().__init__(name)
        self.model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Returning probability for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]

# ===== File: src/models/deep.py =====
# src/models/deep.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # ensure compatibility with transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import tf_keras as tfk   # use separate keras package if needed
from src.models.base_model import BaseModel


class BERTClassifier(BaseModel):
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english",
                 max_len=128, name="bert", from_pt=True):
        super().__init__(name)

        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Mixed precision
        tfk.mixed_precision.set_global_policy("mixed_float16")

        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, from_pt=from_pt
        )
        self._compile_model(lr=2e-5)

    def save(self, path: str):
        """Save pretrained model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, name="bert_loaded"):
        """Load from saved directory."""
        return cls(model_name=path, name=name, from_pt=False)

    def _compile_model(self, lr):
        self.model.compile(
            optimizer=tfk.mixed_precision.LossScaleOptimizer(
                tfk.optimizers.Adam(learning_rate=lr)
            ),
            loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def freeze_backbone(self, num_layers_to_freeze=4):
        """Freeze first N transformer layers."""
        for layer in self.model.layers[0].transformer.layer[:num_layers_to_freeze]:
            layer.trainable = False

# Inside src/models/deep.py, update your train() definition:

    def train(self, X_text, y, epochs=3, batch_size=16, validation_split=0.2, lr=None, patience=3):
        if lr:
            self._compile_model(lr)

        X_train, X_val, y_train, y_val = train_test_split(
            np.array(X_text), np.array(y),
            test_size=validation_split, stratify=y, random_state=42
        )

        train_enc = self.tokenizer(list(X_train), truncation=True, max_length=self.max_len, padding=False)
        val_enc = self.tokenizer(list(X_val), truncation=True, max_length=self.max_len, padding=False)

        def make_gen(enc, labels):
            def gen():
                for i in range(len(labels)):
                    yield ({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}, labels[i])
            return gen

        output_signature = (
            {"input_ids": tf.TensorSpec((None,), tf.int32), "attention_mask": tf.TensorSpec((None,), tf.int32)},
            tf.TensorSpec((), tf.int32),
        )

        train_ds = tf.data.Dataset.from_generator(make_gen(train_enc, y_train), output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(make_gen(val_enc, y_val), output_signature=output_signature)

        train_ds = train_ds.shuffle(buffer_size=min(len(X_train), 1000)).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # --- DYNAMIC CALLBACKS ---
        callbacks = [
            tfk.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=patience,               # Uses the argument
                restore_best_weights=True
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5,                      # Gentler drop 
                patience=max(1, patience - 1),   # Drop LR just before stopping
                min_lr=1e-7
            ),
        ]

        return self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
        )

    def predict_proba(self, X_text):
        tok = self.tokenizer(
            list(X_text), padding=True, truncation=True,
            max_length=self.max_len, return_tensors="tf"
        )
        logits = self.model.predict(
            {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
            },
            verbose=0,
        ).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return probs[:, 1]

# ===== File: src/models/ensemble.py =====
# src/models/ensemble.py
import numpy as np
from src.models.base_model import BaseModel

def safe_binary_probs(probs):
    """
    Ensures probabilities are a 1D array of positive-class probabilities.
    Handles Sklearn (N, 2) and TF/Keras (N, 1) or (N,) shapes.
    """
    if probs.ndim == 2:
        # If shape is (N, 2), take the second column (positive class)
        # If shape is (N, 1), flatten it
        return probs[:, 1] if probs.shape[1] > 1 else probs.ravel()
    return probs

class EnsembleClassifier(BaseModel):
    def __init__(self, models_dict, delegation_threshold=0.3, name="Cascade_Ensemble", specialist_weight=1.0):
        """
        Parameters:
        -----------
        models_dict : dict
            Contains "coarse" (SVM/LogReg) and "fine" (BERT) model instances.
        delegation_threshold : float
            The certainty threshold (e.g., 0.3 means delegating if 0.3 < p < 0.7).
        specialist_weight : float
            Blending weight. Default 1.0 means BERT completely replaces SVM predictions
            on delegated samples.
        """
        super().__init__(name)
        self.models = models_dict
        self.lower = delegation_threshold
        self.upper = 1.0 - delegation_threshold
        self.specialist_weight = specialist_weight

    def train(self, X_sets, y):
        """
        In this pipeline, sub-models are trained independently via specialized scripts.
        """
        raise NotImplementedError("Train sub-models using train_bert.py or train_specialist.py first.")

    def predict_proba(self, X_sets):
        """
        Routes samples based on coarse model certainty.
        
        Parameters:
        -----------
        X_sets : dict
            {
                "coarse": sparse features/TF-IDF for SVM,
                "fine": raw text list for BERT
            }
        """
        # 1. Get initial predictions from Coarse Model (SVM)
        # Check if it's a wrapped Scikit-Learn model or direct
        coarse_model = self.models["coarse"]
        if hasattr(coarse_model, "predict_proba"):
            p_coarse_raw = coarse_model.predict_proba(X_sets["coarse"])
        else:
            p_coarse_raw = coarse_model.model.predict_proba(X_sets["coarse"])
            
        p_coarse = safe_binary_probs(p_coarse_raw)
        final_probs = p_coarse.copy()

        # 2. Identify uncertain samples (Delegation Zone)
        # Unified logic: Delegate if p is between lower and upper threshold
        uncertain_mask = (p_coarse >= self.lower) & (p_coarse <= self.upper)
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) > 0:
            # 3. Run Fine Model (BERT) ONLY on the uncertain samples
            fine_inputs = [X_sets["fine"][i] for i in uncertain_indices]
            p_fine_raw = self.models["fine"].predict_proba(fine_inputs)
            p_fine = safe_binary_probs(p_fine_raw)

            # 4. Sequential Replacement / Blending
            # If specialist_weight = 1.0, SVM is ignored for these samples
            w = self.specialist_weight
            final_probs[uncertain_indices] = (w * p_fine) + ((1 - w) * p_coarse[uncertain_indices])

        return final_probs

    def predict_label(self, X_sets):
        """Returns 0 or 1 based on final blended probabilities."""
        probs = self.predict_proba(X_sets)
        return (probs > 0.5).astype(int)

# ===== File: src/utils/metrics.py =====
import pandas as pd
from src.utils.paths import RESULTS_DIR

def save_to_central_csv(results):
    """
    Saves a dictionary or list of dictionaries containing experiment metrics 
    to a central CSV file, replacing older runs of the same configuration.
    """
    csv_path = RESULTS_DIR / "experiment_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
        
    df_new = pd.DataFrame(results)
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
        
        subset_cols = ['model', 'ngram_range', 'n_concepts', 'z_threshold', 'sentiment_weight']
        
        for col in subset_cols:
            if col not in df.columns:
                df[col] = "N/A"
        
        df[subset_cols] = df[subset_cols].astype(str)
        df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
    else:
        df = df_new
        
    df.to_csv(csv_path, index=False)
    print(f"Metrics successfully saved to {csv_path}")

# ===== File: src/utils/nlp.py =====
from collections import Counter
from tqdm.auto import tqdm
from src.utils.loader import DataLoader 

def process_evaluation_set(
    dataset, 
    set_name, 
    train_mapping, 
    stop_units_set, 
    min_freq, 
    important_set, 
    n_gram_range=(1, 3),
    extractor_obj=None,
    cluster_centers=None,
    printing=True
):
    """
    Filters and maps IDs.
    """
    if printing: print(f"\n1/3: Scanning {set_name} for local rare tokens...")

    set_counts = Counter()
    for item in tqdm(dataset, desc=f"Scanning {set_name}", disable=not printing):
        units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)
        valid_units = [u for u in units if u not in stop_units_set]
        set_counts.update(valid_units)
    
    # These are tokens that appear enough times in the current set to be considered
    significant_units = {u for u, count in set_counts.items() if count >= min_freq}
    
    # Calculate how many of these are actually "new" (Unknown)
    all_unknown_units = sorted(list({u for u in significant_units if u not in train_mapping}))
    num_significant = len(significant_units)
    num_unknown = len(all_unknown_units)
    unknown_pct = (num_unknown / num_significant * 100) if num_significant > 0 else 0

    if printing: print(f"2/3: Mapping {num_unknown} unknown units ({unknown_pct:.1f}% of significant) for {set_name}...")
    
    unknown_mapping = {}
    if extractor_obj and cluster_centers is not None and all_unknown_units: 
        unknown_mapping = extractor_obj.map_units_to_clusters(all_unknown_units, cluster_centers)

    if printing: print(f"3/3: Mapping and Filtering {set_name}...")
        
    stats = {"from_train": 0, "from_unknown": 0, "total_filtered_out": 0}

    for item in tqdm(dataset, desc=f"Processing {set_name}", disable=not printing):
        units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)
        filtered_ids = []
        
        for u in units:
            if u in stop_units_set or u not in significant_units:
                stats["total_filtered_out"] += 1
                continue
            
            cid = None
            is_new = False
            
            if u in train_mapping:
                cid = train_mapping[u]
            elif u in unknown_mapping:
                cid = unknown_mapping[u]
                is_new = True
            
            if cid is not None and cid in important_set:
                filtered_ids.append(cid)
                if is_new: stats["from_unknown"] += 1
                else: stats["from_train"] += 1
            else:
                stats["total_filtered_out"] += 1
        
        item['important_ids'] = filtered_ids

    if printing:
        total_mapped = stats["from_train"] + stats["from_unknown"]
        print(f"\n--- Mapping Verification for {set_name} ---")
        print(f"Significant Vocabulary Discovery:")
        print(f"  - Total Significant Units: {num_significant}")
        print(f"  - Known (from train):     {num_significant - num_unknown}")
        print(f"  - Unknown (New):          {num_unknown} ({unknown_pct:.1f}%)")
        
        if total_mapped > 0:
            print(f"\nToken-Level Impact (Instances in Text):")
            print(f"  - Units from Train Mapping:   {stats['from_train']} ({(stats['from_train']/total_mapped)*100:.1f}%)")
            print(f"  - Units from Unknown Mapping: {stats['from_unknown']} ({(stats['from_unknown']/total_mapped)*100:.1f}%)")
            print(f"  - Successfully re-mapped:     {len(unknown_mapping)}/{num_unknown} unique unknown tokens.")
        else:
            print(f"\n--- Warning: No units from {set_name} were mapped to important concepts ---")

# ===== File: src/utils/paths.py =====
# src/utils/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MLRUNS_DIR = RESULTS_DIR / "mlruns"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "Sentiment"

def get_figure_path(subdir: str, filename: str) -> Path:
    path = FIGURES_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_data_path(subdir: str, filename: str) -> Path:
    path = DATA_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_results_path(subdir: str, filename: str) -> Path:
    path = RESULTS_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_mlflow_uri() -> str:
    """Return absolute file URI for MLflow tracking."""
    return f"file://{MLRUNS_DIR.absolute()}"

# ===== File: src/utils/visualizer.py =====
# src/utils/visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from collections import Counter
from tqdm.auto import tqdm
from IPython.display import display
from typing import List, Dict, Any


class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(title, fontsize=14, pad=15)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_certainty_histogram(preds, probs, y_true, lower=0.3, title="Certainty Distribution"):
        """Density histogram showing model certainty regions."""
        upper = 1 - lower
        correctness = np.where(preds == y_true, "Correct", "Incorrect")

        df = pd.DataFrame({'Probability': probs, 'Result': correctness})
        plt.figure(figsize=(12, 8))

        sns.histplot(data=df, x='Probability', hue='Result', bins=40,
                     multiple="stack", palette={'Correct': '#2ecc71', 'Incorrect': '#e74c3c'},
                     kde=True, alpha=0.7)

        plt.axvline(lower, color='black', linestyle='--', alpha=0.5)
        plt.axvline(upper, color='black', linestyle='--', alpha=0.5)
        plt.title(title, fontsize=14)
        plt.xlabel("Probability of Positive Class")
        plt.ylabel("Number of Samples")
        sns.despine()
        plt.show()

    @staticmethod
    def plot_top_concepts(sf, concept_units, top_n=15):
        """Lollipop plot for the most significant concepts (Z-score)."""
        df_pos = sf.logodds_per_class[1].sort_values("zscore", ascending=False).head(top_n)
        df_neg = sf.logodds_per_class[0].sort_values("zscore", ascending=False).head(top_n)

        df_plot = pd.DataFrame({
            "concept_name": [concept_units[c] for c in df_pos["concept"]] +
                            [concept_units[c] for c in df_neg["concept"]],
            "score": list(df_pos["zscore"]) + list(-df_neg["zscore"]),
            "sentiment": ["Positive"] * top_n + ["Negative"] * top_n
        }).sort_values("score")

        plt.figure(figsize=(12, 8))
        colors = {"Positive": "#2ecc71", "Negative": "#e74c3c"}

        plt.hlines(y=df_plot["concept_name"], xmin=0, xmax=df_plot["score"],
                   color=[colors[s] for s in df_plot["sentiment"]], alpha=0.5)

        for sentiment, color in colors.items():
            mask = df_plot["sentiment"] == sentiment
            plt.scatter(df_plot.loc[mask, "score"], df_plot.loc[mask, "concept_name"],
                        color=color, s=100, label=sentiment, edgecolors='white', zorder=3)

        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f"Top {top_n} Discriminative Concepts (Z-score)", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sentiment_wordclouds(train, train_map, unique_units_to_train, sf, top_n=4, max_words=50, n_gram_range=(1,3)):
        """
        Generates a grid of WordClouds for top positive and negative sentiment concepts.
        Each unit (concept) is treated as an indivisible phrase.
        """
        # 1. Grouping units by Concept ID
        cluster_to_units = {}

        for item in tqdm(train, desc="Building WordCloud clusters"):
            from src.utils.loader import DataLoader
            row_units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)

            for u in row_units:
                if u in train_map and u in unique_units_to_train:
                    cid = train_map[u]
                    normalized_unit = u.replace(" ", "_")
                    cluster_to_units.setdefault(cid, []).append(normalized_unit)

        # 2. Choosing concepts based on Z-score
        pos_ids = (
            sf.logodds_per_class[1]
            .sort_values('zscore', ascending=False)
            .head(top_n)['concept']
            .astype(int)
            .tolist()
        )

        neg_ids = (
            sf.logodds_per_class[1]
            .sort_values('zscore', ascending=True)
            .head(top_n)['concept']
            .astype(int)
            .tolist()
        )

        # 3. Preparing the plot grid
        fig, axes = plt.subplots(2, top_n, figsize=(12, 8))

        def draw_wc(ax, cid, colormap, label):
            # Count frequency of phrases in the cluster
            word_freq = Counter(cluster_to_units.get(cid, []))

            wc = WordCloud(
                background_color='white',
                colormap=colormap,
                max_words=max_words,
                width=400,
                height=300,
                regexp=r"\w+"  # allows underscores
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation='bilinear')

            z_score = sf.logodds_per_class[1].loc[cid, 'zscore']
            ax.set_title(
                f"{label} (ID: {cid})\nZ-score: {z_score:.2f}",
                fontsize=12,
                fontweight='bold'
            )
            ax.axis('off')

        # 4. Drawing positive concepts
        for i, cid in enumerate(pos_ids):
            draw_wc(axes[0, i], cid, 'Greens', "POSITIVE")

        # 5. Drawing negative concepts
        for i, cid in enumerate(neg_ids):
            draw_wc(axes[1, i], cid, 'Reds', "NEGATIVE")

        plt.suptitle(
            "Semantic Analysis of Top Sentiment Concepts",
            fontsize=20,
            y=1.02
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_concept_wordcloud(concept_units, title="Global Map of Discovered Concepts"):
        """
        Visualizes representatives of all clusters.
        Each concept is treated as one indivisible phrase.
        """
        # Join phrases using underscores so that WordCloud treats
        # "bad acting" as one token "bad_acting"
        text = " ".join([u.replace(" ", '_') for u in concept_units])

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="tab20",  # Colorful palette for diverse topics
            max_font_size=100,
            random_state=42,
            regexp=r"\w+"
        ).generate(text)

        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(title, fontsize=18, pad=20, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_extreme_errors(data_list: List[Dict], top_n: int = 5):
        """
        Displays reviews where the model was most confident but incorrect.
        """
        errors = [item for item in data_list if item['sentiment'] != item['pred']]
        errors.sort(key=lambda x: x['prob'], reverse=True)

        print(f"\n--- Top {top_n} 'Confident' Positive Errors (Sure it was 1, actually 0) ---")
        display(pd.DataFrame(errors[:top_n])[['clean_review', 'sentiment', 'pred', 'prob']])

        print(f"\n--- Top {top_n} 'Confident' Negative Errors (Sure it was 0, actually 1) ---")
        display(pd.DataFrame(errors[-top_n:])[['clean_review', 'sentiment', 'pred', 'prob']])

    @staticmethod
    def display_uncertain_errors(data_list: List[Dict], top_n: int = 10):
        """
        Displays reviews where the model was most uncertain (probability near 0.5).
        """
        errors = [item for item in data_list if item['sentiment'] != item['pred']]
        errors.sort(key=lambda x: abs(x['prob'] - 0.5))

        print(f"\n--- Top {top_n} Most Uncertain Reviews (Prob near 0.5) ---")
        df_display = pd.DataFrame(errors[:top_n]).copy()
        df_display['uncertainty_score'] = (df_display['prob'] - 0.5).abs()
        display(df_display[['clean_review', 'sentiment', 'pred', 'prob', 'uncertainty_score']])

    @staticmethod
    def display_dataset_previews(train, val, test, n_rows=5):
        """
        Displays a summary and first few rows of each dataset split.
        """
        data_sets = [("TRAIN", train), ("VALIDATION", val), ("TEST", test)]

        for name, ds in data_sets:
            print(f"\n--- {name} SET (Total: {len(ds)} reviews) ---")
            display(pd.DataFrame(ds[:n_rows]))

