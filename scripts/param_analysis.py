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