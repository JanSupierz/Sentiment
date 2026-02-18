#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

from src.utils.paths import RESULTS_DIR, CONFIGS_DIR, FIGURES_DIR

# Set a clean, academic theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def sweep_exact_cascade(svm_df, bert_preds, y_true):
    """
    Sweeps through certainty requirements to find the exact Local Accuracy 
    AND the True Global Hybrid Accuracy for the whole system.
    """
    # FIX: Start exactly at 50% and go to exactly 100%
    certainty_thresholds = np.linspace(0.500, 1.0, 200)
    svm_probs = svm_df['probability'].values
    svm_preds = (svm_probs > 0.5).astype(int)
    
    results = []
    for c in certainty_thresholds:
        lower_bound = 1.0 - c
        
        # FIX: Force 100% delegation at exactly 1.0 certainty
        if c >= 1.0:
            certain_mask = np.zeros_like(svm_probs, dtype=bool)
        else:
            certain_mask = (svm_probs >= c) | (svm_probs <= lower_bound)
            
        delegated_mask = ~certain_mask
        
        cov = certain_mask.mean()
        delegated = 1.0 - cov 
        
        # 1. LOCAL ACCURACY (What the SVM scores on its retained portion)
        if cov > 0:
            local_acc = (svm_preds[certain_mask] == y_true[certain_mask]).mean()
        else:
            local_acc = 1.0 
            
        # 2. GLOBAL HYBRID ACCURACY (The true performance of the pipeline)
        svm_correct = (svm_preds[certain_mask] == y_true[certain_mask]).sum()
        bert_correct = (bert_preds[delegated_mask] == y_true[delegated_mask]).sum()
        hybrid_acc = (svm_correct + bert_correct) / len(y_true)
            
        results.append((c, cov, delegated, local_acc, hybrid_acc))
        
    return pd.DataFrame(results, columns=['Certainty_Threshold', 'Coverage', 'Delegated', 'Local_Accuracy', 'Hybrid_Accuracy'])

def main():
    raw_dir = RESULTS_DIR / "val" / "raw_predictions"
    fig_dir = FIGURES_DIR / "analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print("❌ No raw predictions found. Run grid_search.py first.")
        return

    # 1. Load Ground Truth and BERT Baseline exact predictions
    bert_file = raw_dir / "bert_basic_baseline.csv"
    if not bert_file.exists():
        print("❌ BERT baseline CSV not found. Run train_bert.py first.")
        return
        
    bert_df = pd.read_csv(bert_file)
    y_true = bert_df['true_label'].values
    bert_preds = (bert_df['probability'].values > 0.5).astype(int)
    bert_global_acc = (bert_preds == y_true).mean()

    csv_files = [f for f in raw_dir.glob("*.csv") if f.name != "bert_basic_baseline.csv"]
    
    all_results = []
    print(f"\n🔍 Sweeping Certainty Thresholds for Exact Hybrid Optimization...")
    
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

    # ==========================================
    # FIND OVERALL WINNER (Based on max Hybrid Accuracy)
    # ==========================================
    best_row = curve_df.loc[curve_df['Hybrid_Accuracy'].idxmax()]
    
    winner_name = best_row['Model']
    best_certainty = best_row['Certainty_Threshold']
    best_delegated = best_row['Delegated']
    best_hybrid_acc = best_row['Hybrid_Accuracy']
    lower_bound = 1.0 - best_certainty

    print(f"\n🏆 CASCADE OPTIMIZATION WINNER: {winner_name}")
    print(f"-> Peak Global System Accuracy: {best_hybrid_acc:.2%} (Beats Base BERT by {best_hybrid_acc - bert_global_acc:+.2%})")
    print(f"-> Required Model Certainty: {best_certainty:.1%} (Accepts probs >= {best_certainty:.3f} or <= {lower_bound:.3f})")
    print(f"-> Data Delegated to BERT: {best_delegated:.2%} (SVM handles {(1-best_delegated):.2%})")

    # Save Best Config
    parts = winner_name.split('_')
    base_m = parts[0] + "_" + parts[1] if parts[0] == "linear" else parts[0]
    ng_str = [p for p in parts if p.startswith('ng')][0].replace('ng', '').split('-')
    k_val = int([p for p in parts if p.startswith('k')][0].replace('k', ''))
    w_val = float([p for p in parts if p.startswith('w')][0].replace('w', ''))
    z_val = float([p for p in parts if p.startswith('z')][0].replace('z', ''))

    best_params = {
        "model": base_m,
        "features": {"ngram_range": [int(ng_str[0]), int(ng_str[1])], "n_concepts": k_val, 
                     "z_threshold": z_val, "use_concepts": k_val > 0, "sentiment_weight": w_val},
        "cascade": {"delegation_threshold": round(float(lower_bound), 3)},
        "models": {base_m: {"C": 1.0}}
    }
    with open(CONFIGS_DIR / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f)

    # ==========================================
    # PLOTTING THE 3 GRAPHS
    # ==========================================
    top_svms = curve_df.groupby('Model')['Hybrid_Accuracy'].max().sort_values(ascending=False).head(4).index.tolist()
    
    plot_df = curve_df[curve_df['Model'].isin(top_svms)].copy()
    plot_df['Model'] = plot_df['Model'].apply(lambda x: x.replace("linear_svm", "SVM").replace("logreg", "LR"))
    clean_top_svms = [m.replace("linear_svm", "SVM").replace("logreg", "LR") for m in top_svms]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Cascade Pipeline Analysis & Global System Optimization", fontsize=20, fontweight='bold', y=1.05)

    colors = sns.color_palette("tab10", len(clean_top_svms))
    palette = {m: c for m, c in zip(clean_top_svms, colors)}

    # --- Graph 1: Local Accuracy vs Certainty Requirement ---
    sns.lineplot(data=plot_df, x="Certainty_Threshold", y="Local_Accuracy", hue="Model", 
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
    sns.lineplot(data=plot_df, x="Certainty_Threshold", y="Delegated", hue="Model", 
                 palette=palette, ax=ax2, linewidth=2.5, legend=False)
    ax2.set_title("2. Workload Management (Deferral Curve)", fontsize=14)
    ax2.set_xlabel(r"Model Prediction Certainty (%)", fontsize=12)
    ax2.set_ylabel("Data Delegated to BERT (%)", fontsize=12)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(0.0, 1.0)

    # --- Graph 3: True Hybrid Optimization Score ---
    sns.lineplot(data=plot_df, x="Delegated", y="Hybrid_Accuracy", hue="Model", 
                 palette=palette, ax=ax3, linewidth=2.5)
    
    # Draw BERT's baseline as a flat line to beat
    ax3.axhline(bert_global_acc, color='black', linestyle='--', linewidth=2, label=f"BERT Global Baseline ({bert_global_acc:.1%})")
    
    # Mark the winning peak!
    ax3.plot(best_delegated, best_hybrid_acc, marker='*', markersize=18, color='gold', markeredgecolor='black', zorder=10, label="Optimal Peak")
    
    ax3.set_title("3. Global System Optimization Score", fontsize=14)
    ax3.set_xlabel("Data Delegated to BERT (%)", fontsize=12)
    ax3.set_ylabel("Combined Cascade Accuracy (%)", fontsize=12)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.legend(fontsize='11', loc='upper right')
    ax3.set_xlim(0.0, 1.0)
    
    plt.tight_layout()
    plot_path = fig_dir / "accuracy_coverage_tradeoff.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved Trade-off graphs to: {plot_path}")

    try:
        from IPython.display import display, Image
        display(Image(filename=plot_path))
    except ImportError:
        pass

if __name__ == "__main__":
    main()