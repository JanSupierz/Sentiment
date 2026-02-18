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