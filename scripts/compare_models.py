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