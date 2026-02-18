# ===== File: Experiment.ipynb =====
import os
import warnings
import logging
import sys
import yaml
import subprocess
from pathlib import Path
from IPython.display import Image, display

# 1. Suppress TensorFlow C++ logs 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 2. Suppress Python Warnings
warnings.filterwarnings("ignore")

# 3. Suppress internal Abseil/TensorFlow Python logging
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Safe import of TensorFlow (must be after env vars)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Set PYTHONPATH to project root so we can import 'src'
project_root = os.getcwd()
os.environ["PYTHONPATH"] = project_root
print(f"PYTHONPATH set to: {project_root}")

config_path = "configs/default.yaml"

with open(config_path) as f:
    base_cfg = yaml.safe_load(f)

print(f"Keys: {base_cfg}")

print("--- Data Preprocessing ---")
subprocess.run(["python", "-m", "src.utils.loader", "--config", config_path], check=True)

# 2. Build Feature Factory
!python -m scripts.build_features --config configs/default.yaml

import pickle
from src.utils.paths import DATA_DIR

def inspect_stop_words(ngram_key="1_3", display_limit=100):
    """Loads and prints stop words, then safely drops them from memory."""
    vocab_path = DATA_DIR / "vocab" / f"vocab_{ngram_key}.pkl"
    
    if not vocab_path.exists():
        print(f"Vocab file not found at {vocab_path}. Run feature building first.")
        return
        
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
        
    stop_units = vocab_data.get("stop_units", [])
    
    print(f"Total dynamically generated stop units: {len(stop_units)}")
    print("-" * 50)
    print(stop_units[:display_limit])

# Run it
inspect_stop_words()

print("\n--- Visualizing Unigrams (Baseline) ---")
!python -m scripts.visualize_sentiment --nmin 1 --nmax 1 --n_concepts 0 --sentiment_weight 0
display(Image("results/figures/vocabulary/ngram_1_1_k0_w0_importance.png"))
!python -m scripts.visualize_sentiment --nmin 1 --nmax 3 --n_concepts 0 --sentiment_weight 0
display(Image("results/figures/vocabulary/ngram_1_3_k0_w0_importance.png"))

print("\n--- Visualizing Semantic Concepts (Clustered) ---")
from IPython.display import Image, display

print("\n=== COMPARISON: STANDARD vs. AUGMENTED CLUSTERING ===")
for weight in [0, 30]:
    print(f"\nAnalyzing Cluster Specialization for Weight {weight}...")
    !python -m scripts.visualize_sentiment \
        --nmin 1 --nmax 3 --n_concepts 5000 --sentiment_weight {weight} --top_n 5

display(Image("results/figures/vocabulary/top_5_Positive_w0.png"))
display(Image("results/figures/vocabulary/top_5_Positive_w30.png"))

display(Image("results/figures/vocabulary/top_5_Negative_w0.png"))
display(Image("results/figures/vocabulary/top_5_Negative_w30.png"))

# 4. Run Grid Evaluation
!python -m scripts.grid_search --config configs/default.yaml --workers 1

# 3. Train BERT Baseline
!python -m scripts.train_bert --config configs/default.yaml

# 5. Find Optimal Threshold & Save Best Params
!python -m scripts.analyse_results
display(Image("results/figures/analysis/accuracy_coverage_tradeoff.png"))

!python -m scripts.param_analysis
display(Image("results/figures/analysis/zscore_full_trellis.png"))

print("\n--- Model Comparison: Baseline vs Clustered Concepts ---")
!python -m scripts.compare_models
display(Image("results/figures/analysis/champion_confusion_matrices.png"))

print("--- Step 6: Train Specialist BERT (Transfer Learning) ---")
!python -m scripts.train_specialist --config configs/default.yaml --best_params configs/best_params.yaml

print("--- Step 7: Run Ensemble (Cascade Evaluation) ---")
!python -m scripts.run_ensemble --config configs/default.yaml --best_params configs/best_params.yaml

!python -m scripts.visualize_thesis
display(Image("results/figures/thesis/thesis_final_results.png"))




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
data:
    train_size: 0.5
    test_size: 0.3
    min_term_freq: 10
    max_df_ratio: 0.4

cascade:  
    specialist_weight: 0.7       

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
    ngram_range: [[1,1], [1,2], [1,3]]
    n_concepts: [0, 1000, 5000, 7000]
    z_threshold: [0, 1.0, 1.96, 2.576]
    sentiment_weight: [0.0, 30.0]
    C: [1.0]

# ===== File: scripts/analyse_results.py =====
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

# ===== File: scripts/build_features.py =====
#!/usr/bin/env python3
import argparse
import yaml
from tqdm import tqdm
from src.utils.paths import PROJECT_ROOT
from src.features import builder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    grid = cfg['grid_search']

    print("--- PHASE 1: Building Base Vocabulary & Embeddings ---")
    unique_ngrams = grid['ngram_range']
    
    for nr in tqdm(unique_ngrams, desc="Base Setup", dynamic_ncols=True):
        builder.build_ngram_index(cfg, nr, splits=['train', 'val'])
        builder.compute_and_cache_embeddings(cfg, nr)

    print("\n--- PHASE 2: Extracting Concepts & Pre-building Matrices ---")
    tasks = []
    for nr in grid['ngram_range']:
        for nc in grid['n_concepts']:
            weights = [0.0] if nc == 0 else grid['sentiment_weight']
            for w in weights:
                # Deduplicate baseline tasks
                if nc == 0 and w > 0:
                    continue
                tasks.append((tuple(nr), nc, w))

    # Remove any stray duplicates
    tasks = list(dict.fromkeys(tasks))

    with tqdm(total=len(tasks), desc="Processing Configurations", dynamic_ncols=True) as pbar:
        for nr, nc, w in tasks:
            pbar.set_description(f"ng{nr[0]}-{nr[1]} | k{nc} | w{int(w)}")
            
            if nc > 0:
                # 1. Cluster embeddings into concepts
                builder.run_extraction_logic(nr, nc, w)
                
                # 2. Pre-build concept matrices (resolves unseen validation words now)
                builder.build_concept_matrices(cfg, nr, nc, w, splits=['train', 'val'])
            
            # 3. Compute log-odds stats (relies on the train matrix)
            builder.run_stats_logic(cfg, nr, nc, w)
            
            pbar.update(1)

    print("\nFeature Factory Complete! Matrices cached. Ready for parallel grid search.")

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
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfTransformer
import dask
from dask.distributed import Client, LocalCluster, as_completed
from tqdm import tqdm

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.features.concept_remap import remap_sparse_matrix
from src.utils.paths import PROJECT_ROOT, DATA_DIR
from src.features.builder import load_representation

def run_grid_iter(static_cfg, nr, nc, w, z, m):
    # Construct a unique, descriptive name
    run_id = f"{m}_ng{nr[0]}-{nr[1]}_k{nc}_w{int(w)}_z{z}"
    
    try:
        X_train, y_train = load_representation(static_cfg, nr, nc, w, z, 'train')
        X_val, y_val = load_representation(static_cfg, nr, nc, w, z, 'val')
        
        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)
        
        if m == 'linear_svm':
            model = LinearSVMClassifier(C=static_cfg['grid_search']['C'][0], name=run_id)
        else:
            model = LogisticRegressionClassifier(C=static_cfg['grid_search']['C'][0], name=run_id)
            
        model.train(X_train_tfidf, y_train)
        model.evaluate(X_val_tfidf, y_val, name="val")
        
        return f"Done: {run_id}"
    except Exception as e:
        return f"Failed {run_id}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        static_cfg = yaml.safe_load(f)
    grid = static_cfg['grid_search']

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    delayed_tasks = []
    for nr, nc, z, w, m in itertools.product(
        grid['ngram_range'], grid['n_concepts'], grid['z_threshold'], 
        grid['sentiment_weight'], grid['models']
    ):
        if nc == 0 and w > 0: continue # Deduplicate
        delayed_tasks.append(dask.delayed(run_grid_iter)(static_cfg, nr, nc, w, z, m))
    
    if delayed_tasks:
        print(f"🚀 Starting Grid Search with {len(delayed_tasks)} iterations...")
        futures = client.compute(delayed_tasks)
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search Progress"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"Failed: {e}")

        failed = [r for r in results if r.startswith("Failed")]
        print(f"\n✅ Completed: {len(results) - len(failed)}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
            for f in failed[:5]: print(f"  {f}")

    client.close()

if __name__ == "__main__":
    main()

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

    X_train_txt = df_train['clean_review'].tolist()
    y_train = df_train['sentiment'].tolist()
    X_val_txt = df_val['clean_review'].tolist()
    y_val = df_val['sentiment'].values.astype(np.int32)

    bert_cfg = cfg['bert']['basic']
    
    # Init with a descriptive name. BaseModel uses this to save the raw_predictions CSV
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

    # Automatically saves to results/val/raw_predictions/bert_basic_baseline.csv
    model.evaluate(X_val_txt, y_val, name="val")
    
    # FIX: Explicitly define the destination variable so the print statement works
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
import joblib
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
    (RESULTS_DIR / "test" / "raw_predictions").mkdir(parents=True, exist_ok=True)

    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']

    print("Loading data...")
    X_train_sp, y_train = load_representation(cfg, nr, nc, w, z, 'train')
    X_val_sp, y_val = load_representation(cfg, nr, nc, w, z, 'val')
    X_test_sp, y_test = load_representation(cfg, nr, nc, w, z, 'test')

    df_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")

    X_train_txt = df_train['clean_review'].tolist()
    X_val_txt = df_val['clean_review'].tolist()
    X_test_txt = df_test['clean_review'].tolist()

    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train_sp)
    X_val_tfidf = tfidf.transform(X_val_sp)
    X_test_tfidf = tfidf.transform(X_test_sp)

    # --- SVM ---
    print("\nTraining SVM with best parameters...")
    if cfg['model'] == 'linear_svm':
        svm = LinearSVMClassifier(C=cfg['models']['linear_svm']['C'], name='Base_SVM')
    else:
        svm = LogisticRegressionClassifier(C=cfg['models']['logreg']['C'], name='Base_LogReg')
    svm.train(X_train_tfidf, y_train)

    svm_path = MODELS_DIR / "svm_base"
    svm.save(str(svm_path))
    print(f"SVM saved to {svm_path}.joblib")

    # --- Mine hard samples from validation ---
    threshold = cfg['cascade']['delegation_threshold']
    lower = threshold
    upper = 1.0 - lower

    probs_val = safe_binary_probs(svm.predict_proba(X_val_tfidf))
    mask_uncertain = (probs_val >= lower) & (probs_val <= upper)
    hard_indices = np.where(mask_uncertain)[0]
    print(f"Validation hard samples: {len(hard_indices)} ({len(hard_indices)/len(y_val):.2%})")

    X_hard_txt = [X_val_txt[i] for i in hard_indices]
    y_hard = [y_val[i] for i in hard_indices]

    # --- Load Basic BERT ---
    bert_basic_path = MODELS_DIR / "bert_basic"
    if not bert_basic_path.exists():
        raise FileNotFoundError("Basic BERT model not found. Run scripts/train_bert.py first.")
    bert_basic = BERTClassifier.load(str(bert_basic_path), name="BERT_Basic")

    # --- Train Specialist BERT ---
    print("\nFine‑tuning specialist BERT on hard samples...")
    bert_spec = BERTClassifier.load(str(bert_basic_path), name="BERT_Specialist")
    bert_spec.freeze_backbone(num_layers_to_freeze=4)

    spec_cfg = cfg['bert']['specialist']
    epochs = int(spec_cfg['epochs'])
    batch_size = int(spec_cfg['batch_size'])
    lr = float(spec_cfg['learning_rate'])
    patience = int(spec_cfg['patience'])

    bert_spec.train(
        X_hard_txt, y_hard,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience
    )

    bert_spec_path = MODELS_DIR / "bert_specialist"
    bert_spec.save(str(bert_spec_path))
    print(f"Specialist BERT saved to {bert_spec_path}")

    # --- Evaluate specialist on test set ---
    print("\nEvaluating specialist on test set...")
    probs_spec = safe_binary_probs(bert_spec.predict_proba(X_test_txt))
    preds_spec = (probs_spec > 0.5).astype(int)
    acc_spec = (preds_spec == y_test).mean()
    print(f"Specialist test accuracy: {acc_spec:.4f}")

    eval_df = pd.DataFrame({
        'true_label': y_test,
        'probability': probs_spec
    })
    eval_df.to_csv(RESULTS_DIR / "test" / "raw_predictions" / "bert_specialist.csv", index=False)
    print("Test predictions saved to results/test/raw_predictions/bert_specialist.csv")

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
# src/features/builder.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
import torch
from sentence_transformers import SentenceTransformer
import faiss
faiss.verbose = False
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.loader import DataLoader
from src.features.sentiment import SentimentFeatures
from src.features.concept_remap import remap_sparse_matrix
from src.utils.paths import DATA_DIR

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

def build_ngram_index(cfg, ngram_range, splits=None):
    """
    Build vocabulary and count matrices for specified splits.
    If splits is None, builds all three (train, val, test).
    """
    if splits is None:
        splits = ['train', 'val', 'test']
        
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    vocab_path = DATA_DIR / "vocab" / f"vocab_{key}.pkl"
    cache_dir = DATA_DIR / "cache_matrices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. JIT VOCABULARY BUILDING ---
    if not vocab_path.exists():
        print(f"Vocab missing for {key}. Building from train split...")
        try:
            train_df = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
        except FileNotFoundError:
            raise ValueError("Train split (train.parquet) must be available to build vocabulary.")

        min_freq = cfg['data']['min_term_freq']
        max_df_ratio = cfg['data']['max_df_ratio']

        df_by_label = defaultdict(Counter)
        for _, row in train_df.iterrows():
            units = set(DataLoader.get_ngrams(row["clean_bow"], ngram_range))
            df_by_label[row["sentiment"]].update(units)

        n_docs_label = Counter(train_df["sentiment"])
        stop_sets = []
        for label in [0, 1]:
            thresh = n_docs_label[label] * max_df_ratio
            stop_sets.append({u for u, cnt in df_by_label[label].items() if cnt > thresh})
        stop_units = set.intersection(*stop_sets) if stop_sets else set()

        tf_counts = Counter()
        for _, row in train_df.iterrows():
            units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
            tf_counts.update([u for u in units if u not in stop_units])

        vocab = sorted([u for u, cnt in tf_counts.items() if cnt >= min_freq])
        unit_to_id = {u: i for i, u in enumerate(vocab)}

        DATA_DIR.joinpath("vocab").mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "wb") as f:
            pickle.dump({"vocab": vocab, "unit_to_id": unit_to_id, "stop_units": list(stop_units)}, f)

    # --- 2. JIT SPLIT MATRIX BUILDING ---
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
        
    unit_to_id = vocab_data['unit_to_id']
    stop_units = set(vocab_data['stop_units'])
    vocab_len = len(vocab_data['vocab'])

    for sp in splits:
        cache_file = cache_dir / f"X_{sp}_{key}.npz"
        if not cache_file.exists():
            print(f"Cache missing for {sp} split. Building matrix...")
            df = pd.read_parquet(DATA_DIR / "preprocessed" / f"{sp}.parquet")
            
            rows, cols, data = [], [], []
            for doc_id, (_, row) in enumerate(df.iterrows()):
                units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
                for u in units:
                    if u in stop_units:
                        continue
                    uid = unit_to_id.get(u)
                    if uid is not None:
                        rows.append(doc_id)
                        cols.append(uid)
                        data.append(1)
                        
            X = csr_matrix((data, (rows, cols)), shape=(len(df), vocab_len), dtype=np.float32)
            save_npz(cache_file, X)


def compute_and_cache_embeddings(cfg, ngram_range):
    """Compute SBERT embeddings + VADER sentiment scores for all units (always needed if clustering)."""
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    cache_path = DATA_DIR / "cache_matrices" / f"emb_{key}.npz"
    if cache_path.exists():
        return
        
    with open(DATA_DIR / "vocab" / f"vocab_{key}.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    units = vocab_data['vocab']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = cfg['bert']['sentence_model']
    model = SentenceTransformer(model_name).to(device)
    embeddings = model.encode(units, batch_size=256, convert_to_tensor=True,
                              device=device, show_progress_bar=False)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = np.array([analyzer.polarity_scores(u)['compound'] for u in units],
                                dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(cache_path, embeddings=embeddings, sentiment_scores=sentiment_scores)


def run_extraction_logic(ngram_range, n_concepts, sentiment_weight):
    """Cluster units into concepts (if n_concepts>0) and save centroids."""
    if n_concepts == 0:
        return
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    out_dir = DATA_DIR / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"concepts_{key}_k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if out_path.exists():
        return

    with open(DATA_DIR / "vocab" / f"vocab_{key}.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    units = vocab_data['vocab']

    data = np.load(DATA_DIR / "cache_matrices" / f"emb_{key}.npz")
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
    
    # Save both mapping and centroids
    with open(out_path, "wb") as f:
        pickle.dump({
            'unit_to_cluster': dict(zip(units, labels.flatten().tolist())),
            'centroids': centroids,
            'sentiment_weight': sentiment_weight,
            'ngram_range': ngram_range,
            'n_concepts': actual_k
        }, f)
        

def run_stats_logic(cfg, ngram_range, n_concepts, sentiment_weight):
    """Compute log‑odds statistics for the representation (requires train matrix)."""
    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    repr_key = f"{key}_k{n_concepts}_w{int(sentiment_weight)}" if n_concepts > 0 else f"{key}_raw"
    out_dir = DATA_DIR / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stats_{repr_key}.pkl"
    if out_path.exists():
        return

    # Ensure train matrix exists for this ngram_range
    train_cache = DATA_DIR / "cache_matrices" / f"X_train_{key}.npz"
    if not train_cache.exists():
        build_ngram_index(cfg, ngram_range, splits=['train'])

    X_train = load_npz(train_cache)
    y_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")['sentiment'].values

    if n_concepts > 0:
        with open(DATA_DIR / "concepts" / f"concepts_{key}_k{n_concepts}_w{int(sentiment_weight)}.pkl", "rb") as f:
            c_data = pickle.load(f)
        with open(DATA_DIR / "vocab" / f"vocab_{key}.pkl", "rb") as f:
            v_data = pickle.load(f)
        id_to_cluster = {v_data['unit_to_id'][u]: cid for u, cid in c_data['unit_to_cluster'].items()}
        X_train = remap_sparse_matrix(X_train, id_to_cluster, n_concepts)

    sf = SentimentFeatures()
    sf.fit(X_train, y_train.tolist())
    with open(out_path, "wb") as f:
        pickle.dump(sf.logodds_per_class, f)


def ensure_representation(cfg, ngram_range, n_concepts, sentiment_weight, splits=None):
    """
    Ensure that all required files for a given representation exist for the specified splits.
    If splits is None, builds all three splits.
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    build_ngram_index(cfg, ngram_range, splits)
    compute_and_cache_embeddings(cfg, ngram_range)
    if n_concepts > 0:
        run_extraction_logic(ngram_range, n_concepts, sentiment_weight)
    run_stats_logic(cfg, ngram_range, n_concepts, sentiment_weight)


def map_unseen_units(unseen_units, concept_data_path, model_name='all-MiniLM-L6-v2', batch_size=256):
    """
    Map a list of unit strings (unseen during training) to concept IDs using saved centroids.
    Returns dict {unit: concept_id}.
    """
    # Load centroids and metadata
    with open(concept_data_path, 'rb') as f:
        data = pickle.load(f)
    centroids = data['centroids']
    sentiment_weight = data['sentiment_weight']

    # Get models
    model = _get_sentence_model(model_name)
    analyzer = _get_analyzer()

    # Compute embeddings for unseen units (batched on GPU)
    embeddings = model.encode(unseen_units, batch_size=batch_size, convert_to_tensor=True,
                              device=model.device, show_progress_bar=False)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    sent_scores = np.array([analyzer.polarity_scores(u)['compound'] for u in unseen_units], dtype=np.float32).reshape(-1, 1)
    embeddings = np.hstack([embeddings, sent_scores * sentiment_weight])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build FAISS index on centroids (done once per call, not per word)
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids.astype(np.float32))
    distances, labels = index.search(embeddings.astype(np.float32), 1)

    # Similarity threshold (cosine from L2 on normalized vectors)
    SIM_THRESHOLD = 0.6
    similarities = 1 - distances.flatten() / 2
    mapping = {}
    for u, l, sim in zip(unseen_units, labels.flatten(), similarities):
        if sim >= SIM_THRESHOLD:
            mapping[u] = int(l)
    return mapping


def build_concept_matrices(cfg, ngram_range, n_concepts, sentiment_weight, splits=None):
    """
    Build and save concept-level sparse matrices for the requested splits.
    If splits is None, builds for all three (train, val, test).
    Matrices are saved in data/concept_matrices/.
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    nmin, nmax = ngram_range
    key = f"{nmin}_{nmax}"
    concept_file = DATA_DIR / "concepts" / f"concepts_{key}_k{n_concepts}_w{int(sentiment_weight)}.pkl"
    if not concept_file.exists():
        raise FileNotFoundError(f"Concept file not found: {concept_file}. Run run_extraction_logic first.")

    # Load concept data (centroids + training mapping)
    with open(concept_file, 'rb') as f:
        cdata = pickle.load(f)
    unit_to_cluster_train = cdata['unit_to_cluster']   # mapping for training units only
    centroids = cdata['centroids']
    actual_n_concepts = centroids.shape[0]

    # Load vocabulary (unit_to_id)
    vocab_file = DATA_DIR / "vocab" / f"vocab_{key}.pkl"
    with open(vocab_file, 'rb') as f:
        vdata = pickle.load(f)
    unit_to_id = vdata['unit_to_id']
    id_to_unit = {idx: unit for unit, idx in unit_to_id.items()}

    # Cache for mapping of unseen units (per configuration, reused across splits)
    mapping_cache = {}

    # Preload sentence transformer model name and frequency threshold from cfg
    model_name = cfg['bert']['sentence_model']
    min_freq = cfg['data']['min_term_freq']

    for split in splits:
        out_dir = DATA_DIR / "concept_matrices"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{split}_{key}_k{n_concepts}_w{int(sentiment_weight)}.npz"
        
        if out_file.exists():
            print(f"Concept matrix for {split} already exists: {out_file}")
            continue
            
        print(f"Building concept matrix for {split} (k={n_concepts}, w={sentiment_weight})...")
        df = pd.read_parquet(DATA_DIR / "preprocessed" / f"{split}.parquet")

        if split == 'train':
            # Use the unit matrix and remap via training mapping
            unit_mat = load_npz(DATA_DIR / "cache_matrices" / f"X_train_{key}.npz")
            id_to_cluster = {}
            for uid in range(unit_mat.shape[1]):
                unit = id_to_unit.get(uid)
                if unit is not None and unit in unit_to_cluster_train:
                    id_to_cluster[uid] = unit_to_cluster_train[unit]
                    
            if not id_to_cluster:
                X_concept = csr_matrix((len(df), 0), dtype=np.float32)
            else:
                valid_uids = sorted(id_to_cluster.keys())
                remap = np.array([id_to_cluster[uid] for uid in valid_uids])
                X_unit_subset = unit_mat[:, valid_uids]
                coo = X_unit_subset.tocoo()
                rows, cols, data = coo.row, remap[coo.col], coo.data
                df_agg = pd.DataFrame({'row': rows, 'col': cols, 'data': data}).groupby(['row', 'col'], as_index=False).sum()
                X_concept = csr_matrix((df_agg['data'].values,
                                        (df_agg['row'].values, df_agg['col'].values)),
                                       shape=(len(df), actual_n_concepts))
        else:
            # ===== OPTIMIZED 3-PASS SYSTEM FOR VAL/TEST =====
            
            # Pass 1: Count frequencies of all units in this split
            split_counts = Counter()
            for _, row in df.iterrows():
                units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
                split_counts.update(units)
                
            # Filter for unseen units that meet the minimum frequency threshold
            unseen_set = set()
            for u, count in split_counts.items():
                if count >= min_freq:
                    uid = unit_to_id.get(u)
                    # If unit is completely new, OR in vocab but wasn't clustered
                    if uid is None or id_to_unit.get(uid) not in unit_to_cluster_train:
                        if u not in mapping_cache:
                            unseen_set.add(u)

            # Pass 2: Batch-map ONLY the frequent unseen units
            if unseen_set:
                unseen_list = list(unseen_set)
                print(f"Batch mapping {len(unseen_list)} frequent new units for {split} (dropped rare units)...")
                new_mappings = map_unseen_units(unseen_list, concept_file, model_name=model_name)
                
                for u in unseen_list:
                    mapping_cache[u] = new_mappings.get(u, -1)

            # Pass 3: Build the sparse matrix
            rows, cols, data = [], [], []
            for doc_id, (_, row) in enumerate(df.iterrows()):
                units = DataLoader.get_ngrams(row["clean_bow"], ngram_range)
                for u in units:
                    uid = unit_to_id.get(u)
                    
                    # If it's a known training unit, we always use it
                    if uid is not None and id_to_unit.get(uid) in unit_to_cluster_train:
                        cid = unit_to_cluster_train[id_to_unit[uid]]
                    else:
                        # Otherwise, check the cache (returns -1 if it was rare/ignored or below similarity threshold)
                        cid = mapping_cache.get(u, -1)

                    if cid != -1:
                        rows.append(doc_id)
                        cols.append(cid)
                        data.append(1)

            if len(rows) == 0:
                X_concept = csr_matrix((len(df), 0), dtype=np.float32)
            else:
                X_concept = coo_matrix((data, (rows, cols)),
                                       shape=(len(df), actual_n_concepts)).tocsr()

        save_npz(out_file, X_concept)
        print(f"Saved concept matrix to {out_file}")


def load_representation(cfg, ngram_range, n_concepts, sentiment_weight, z_threshold, split):
    """
    Load the sparse count matrix for a given split after ensuring it exists.
    Applies Z‑score filtering if z_threshold > 0.
    Returns (X_sparse, y_array).
    """
    key = f"{ngram_range[0]}_{ngram_range[1]}"
    y = pd.read_parquet(DATA_DIR / "preprocessed" / f"{split}.parquet")['sentiment'].values

    if n_concepts == 0:
        # Original unit-level representation
        build_ngram_index(cfg, ngram_range, splits=[split])
        X = load_npz(DATA_DIR / "cache_matrices" / f"X_{split}_{key}.npz")
        n_features = X.shape[1]
        repr_key = f"{key}_raw"
    else:
        # Concept-level representation – use precomputed matrices
        concept_dir = DATA_DIR / "concept_matrices"
        concept_file = concept_dir / f"{split}_{key}_k{n_concepts}_w{int(sentiment_weight)}.npz"
        if not concept_file.exists():
            # Build concept matrices for all splits if missing (this will be done only once)
            build_concept_matrices(cfg, ngram_range, n_concepts, sentiment_weight, splits=[split])
        X = load_npz(concept_file)
        n_features = X.shape[1]
        repr_key = f"{key}_k{n_concepts}_w{int(sentiment_weight)}"

    # Z-score filtering
    if z_threshold > 0:
        stats_path = DATA_DIR / "stats" / f"stats_{repr_key}.pkl"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        important_pos = set(stats[1][stats[1]['zscore'] > z_threshold]['concept'])
        important_neg = set(stats[0][stats[0]['zscore'] > z_threshold]['concept'])
        important = important_pos | important_neg
        mask = np.array([i in important for i in range(n_features)])
        X = X[:, mask]

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

# ===== File: src/features/sentiment.py =====
# src/features/sentiment.py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import itertools
from typing import List, Union, Optional


class SentimentFeatures:
    """
    Compute log‑odds and Z‑scores for each feature (concept or unit).
    Can accept both list‑of‑lists (backward compatibility) and CSR matrix.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.logodds_per_class = {}   # dict {0: df, 1: df}
        self.concept_list = None     # list of feature IDs (columns)
        self._feature_to_idx = None

    def fit(self,
            X: Union[List[List[int]], csr_matrix],
            y: List[int],
            feature_names: Optional[List[int]] = None):
        """
        Fit sentiment statistics.
        If X is CSR matrix: shape (n_docs, n_features)
        If X is list of lists: each inner list contains feature IDs (may have duplicates)
        feature_names: only needed if X is list-of-lists (to define column order)
        """
        if isinstance(X, csr_matrix):
            self._fit_sparse(X, y)
        else:
            self._fit_list(X, y, feature_names)
        return self

    def _fit_sparse(self, X: csr_matrix, y: List[int]):
        """Fast path: directly use CSR matrix."""
        n_docs, n_features = X.shape
        self.concept_list = list(range(n_features))   # assume column indices are feature IDs
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

    def _fit_list(self, X: List[List[int]], y: List[int], feature_names: Optional[List[int]] = None):
        """Original list‑of‑lists implementation."""
        unique_concepts = sorted(set(itertools.chain.from_iterable(X)))
        if feature_names is not None:
            # Use provided order (important for consistency)
            unique_concepts = feature_names
        self.concept_list = unique_concepts
        self._feature_to_idx = {c: i for i, c in enumerate(unique_concepts)}

        # Build sparse matrix
        rows, cols, data = [], [], []
        for doc_id, concepts in enumerate(X):
            for c in concepts:
                if c in self._feature_to_idx:
                    rows.append(doc_id)
                    cols.append(self._feature_to_idx[c])
                    data.append(1)   # each occurrence counts
        X_mat = coo_matrix((data, (rows, cols)),
                           shape=(len(X), len(unique_concepts))).tocsr()
        self._fit_sparse(X_mat, y)

    def filter_by_zscore(self, threshold: float) -> set:
        """
        Return set of concept IDs whose Z‑score > threshold for either class.
        """
        pos_set = set(self.logodds_per_class[1][self.logodds_per_class[1]['zscore'] > threshold]['concept'])
        neg_set = set(self.logodds_per_class[0][self.logodds_per_class[0]['zscore'] > threshold]['concept'])
        return pos_set | neg_set

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
        # LinearSVC by default doesn't return probabilities, so we use CalibratedClassifierCV
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

# ===== File: src/utils/loader.py =====
import os
import warnings
import logging
import json
import re
import unicodedata
import contractions
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import argparse
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.object.*")
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.utils.visualizer import ModelVisualizer
from src.utils.paths import DATA_DIR

class DataLoader:
    TRANSLATE_TABLE = str.maketrans({
        "`": "'", "´": "'", "’": "'", "‘": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "-", "−": "-",
        "\x96": "-", "\x97": "-",
        "…": "..."
    })

    @classmethod
    def process_single_item(cls, item: Tuple[bytes, int]) -> Dict:
        """Clean and tokenise a single review."""
        text_raw, label = item
        text_raw = text_raw.decode("utf-8") if isinstance(text_raw, bytes) else text_raw

        text = unicodedata.normalize('NFKC', text_raw)
        text = text.translate(cls.TRANSLATE_TABLE)
        text = re.sub(r'<[^>]+>', ' ', text)
        clean_bert = re.sub(r'\s+', ' ', text).strip()
        clean_bow = contractions.fix(clean_bert.lower())

        return {
            "review": text_raw,
            "clean_review": clean_bert,
            "clean_bow": clean_bow,
            "sentiment": int(label)
        }

    @staticmethod
    def get_ngrams(text: str, ngram_range: Tuple[int, int] = (1, 3)) -> List[str]:
        """Extract n‑grams (word‑level) from pre‑tokenised BoW text."""
        tokens = re.findall(r'\w+|[^\w\s]', text)
        min_n, max_n = ngram_range
        units = []
        for n in range(min_n, max_n + 1):
            if n == 1:
                units.extend(tokens)
            else:
                grams = zip(*[tokens[i:] for i in range(n)])
                units.extend([" ".join(g) for g in grams])
        return units

    @classmethod
    def load_imdb(cls, n_jobs: int = 4, train_size: float = 0.6, test_size: float = 0.8) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load IMDb reviews, clean, deduplicate, split.
        Returns (train, val, test) as lists of dicts.
        """
        save_dir = str(DATA_DIR / "IMDb")

        print("Loading IMDb reviews...")
        ds = tfds.load('imdb_reviews', split='train+test', as_supervised=True)
        raw_data = list(tfds.as_numpy(ds))

        seen = set()
        unique_raw = []
        for t, l in raw_data:
            if t not in seen:
                seen.add(t)
                unique_raw.append((t, l))

        print(f"Found {len(raw_data)-len(unique_raw)} duplicates.")
        print(f"Cleaning {len(unique_raw)} reviews...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            processed = list(tqdm(executor.map(cls.process_single_item, unique_raw), total=len(unique_raw)))

        labels = [item['sentiment'] for item in processed]
        train, temp_data, _, temp_labels = train_test_split(
            processed, labels, test_size=1 - train_size, random_state=42, stratify=labels
        )
        val, test = train_test_split(
            temp_data, test_size=test_size, random_state=42, stratify=temp_labels
        )

        print(f"Loaded dataset with {len(train)} train, {len(val)} val and {len(test)} test samples.")

        os.makedirs(save_dir, exist_ok=True)
        for name, dataset in zip(["train", "val", "test"], [train, val, test]):
            path = os.path.join(save_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Loading done. JSON files saved to {save_dir}")

        return train, val, test

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    train, val, test = DataLoader.load_imdb(
        train_size=cfg['data']['train_size'],
        test_size=cfg['data']['test_size']
    )

    def to_df(data):
        df = pd.DataFrame(data)
        if 'review' in df.columns:
            df.drop(columns=['review'], inplace=True)
        return df

    train_df = to_df(train)
    val_df   = to_df(val)
    test_df  = to_df(test)

    # Save as Parquet
    out_dir = DATA_DIR / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_dir / "train.parquet")
    val_df.to_parquet(out_dir / "val.parquet")
    test_df.to_parquet(out_dir / "test.parquet")

    print(f"Preprocessing done. Parquet files saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)

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

