import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import argparse
import subprocess
import sys
from matplotlib.patches import Patch

# ============================================================
# Global font size for ALL text elements
# ============================================================
FONT_SIZE = 15

# Mapping of internal token names -> display names on the y‑axis
TOKEN_DISPLAY_MAP = {
    'tokens_cased':      'Tokens Cased',
    'tokens_lower':      'Tokens Lower',
    'tokens_letters':    'Tokens Letters',
    'tokens_filtered':   'Tokens Filtered',
    'tokens_stemmed':    'Tokens Stemmed',
    'tokens_lemmatized': 'Tokens Lemmatized'
}

def load_mcnemar_results(mcnemar_files):
    if isinstance(mcnemar_files, (str, Path)):
        mcnemar_files = [mcnemar_files]
    p_dict = {}
    for file in mcnemar_files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            if 'Model' in df.columns:
                comp_col = 'Model'
            elif 'Compared_Model' in df.columns:
                comp_col = 'Compared_Model'
            else:
                comp_col = 'Model_Porównywany'
            comp = row[comp_col]
            base = row['Baseline']
            p = row['P_Value']
            p_dict[(comp, base)] = p
    return p_dict

def get_baseline_filename(model_label, rep):
    model_map_rev = {
        'Linear SVM': 'linear_svm',
        'Logistic Regression': 'logreg'
    }
    model_raw = model_map_rev[model_label]
    return f"text_expanded_lower_{rep}_{model_raw}.csv"

def find_file(filename, search_dirs):
    for directory in search_dirs:
        file_path = directory / filename
        if file_path.exists():
            return file_path
    return None

def run_mcnemar_tests(base_dir="results/val", run_names=None, force=False, z_threshold=2):
    if run_names is None:
        run_names = ["run_on_preprocessing", "run_custom", "run_basic"]
    base_path = Path(base_dir)
    raw_dirs = {run: base_path / run / "raw_predictions" for run in run_names}
    models = ["linear_svm", "logreg"]
    baseline_reps = ["BoW", "TF-IDF"]
    token_columns = list(TOKEN_DISPLAY_MAP.keys())
    
    # Format Z-score cleanly for filenames (e.g., 2 instead of 2.0, but allows 2.5)
    z_val = int(z_threshold) if float(z_threshold).is_integer() else z_threshold
    
    generated_files = []

    for model in models:
        for base_rep in baseline_reps:
            baseline_filename = f"text_expanded_lower_{base_rep}_{model}.csv"
            baseline_path = raw_dirs["run_basic"] / baseline_filename
            if not baseline_path.exists():
                continue

            targets = []
            if base_rep == "BoW":
                for token in token_columns:
                    target_filename = f"{token}_BoW_{model}.csv"
                    target_path = raw_dirs["run_on_preprocessing"] / target_filename
                    if target_path.exists():
                        targets.append(f"run_on_preprocessing:::{target_path.stem}")
                for token in token_columns:
                    target_filename = f"{token}_CustomBow_Z{z_val}_{model}.csv"
                    target_path = raw_dirs["run_custom"] / target_filename
                    if target_path.exists():
                        targets.append(f"run_custom:::{target_path.stem}")
            else:
                for token in token_columns:
                    target_filename = f"{token}_TF-IDF_{model}.csv"
                    target_path = raw_dirs["run_on_preprocessing"] / target_filename
                    if target_path.exists():
                        targets.append(f"run_on_preprocessing:::{target_path.stem}")
                for token in token_columns:
                    target_filename = f"{token}_CustomTfidf_Z{z_val}_{model}.csv"
                    target_path = raw_dirs["run_custom"] / target_filename
                    if target_path.exists():
                        targets.append(f"run_custom:::{target_path.stem}")

            if not targets:
                continue

            out_filename = f"mcnemar_{model}_{base_rep}.csv"
            out_path = Path("results/analysis") / out_filename
            if out_path.exists() and not force:
                generated_files.append(str(out_path))
                continue

            cmd = [
                sys.executable, "-m", "scripts.run_statistical_test",
                "--targets", *targets,
                "--baseline", str(baseline_path.relative_to(Path("results"))),
                "--split", "val",
                "--output", out_filename
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                generated_files.append(str(out_path))

    return generated_files

def plot_ablation_study(base_dir="results/val", run_names=None, mcnemar_files=None, z_threshold=2):
    if run_names is None:
        run_names = ["run_on_preprocessing", "run_custom", "run_basic"]

    base_path = Path(base_dir)
    search_dirs = [base_path / run / "classification_reports" for run in run_names]
    valid_dirs = [d for d in search_dirs if d.exists()]
    if not valid_dirs:
        return

    token_columns = list(TOKEN_DISPLAY_MAP.keys())       # internal names
    display_order = list(TOKEN_DISPLAY_MAP.values())     # display names for y‑axis

    baseline_col = "text_expanded"
    baseline_casing = "lower"
    baseline_reps = ["BoW", "TF-IDF"]
    representations = ["BoW", "TF-IDF", "CustomBow", "CustomTfidf"]
    models = ["linear_svm", "logreg"]
    
    # Format Z-score cleanly for filenames
    z_val = int(z_threshold) if float(z_threshold).is_integer() else z_threshold

    records = []
    baselines = {"Linear SVM": {}, "Logistic Regression": {}}

    for model in models:
        model_label = "Linear SVM" if model == "linear_svm" else "Logistic Regression"
        for rep in baseline_reps:
            base_filename = f"{baseline_col}_{baseline_casing}_{rep}_{model}.csv"
            base_file = find_file(base_filename, valid_dirs)
            if base_file:
                base_df = pd.read_csv(base_file, index_col=0)
                baselines[model_label][rep] = base_df.loc["macro avg", "f1-score"]

        for token_col in token_columns:
            for rep in representations:
                if rep in ["CustomTfidf", "CustomBow"]:
                    filename = f"{token_col}_{rep}_Z{z_val}_{model}.csv"
                else:
                    filename = f"{token_col}_{rep}_{model}.csv"
                filepath = find_file(filename, valid_dirs)
                if filepath:
                    df = pd.read_csv(filepath, index_col=0)
                    records.append({
                        "Token Processing": TOKEN_DISPLAY_MAP[token_col],  # readable name
                        "Representation": rep,
                        "Model": model_label,
                        "Macro F1-Score": df.loc["macro avg", "f1-score"],
                        "Filename": filepath.stem
                    })

    if not records:
        return

    results_df = pd.DataFrame(records)
    global_min_f1 = results_df["Macro F1-Score"].min()
    global_max_f1 = results_df["Macro F1-Score"].max()

    p_dict = {}
    if mcnemar_files:
        p_dict = load_mcnemar_results(mcnemar_files)

    # ---------- Apply global font settings ----------
    sns.set_theme(style="whitegrid")
    
    # FORCING MATPLOTLIB TO USE FONT_SIZE FOR EVERY TEXT ELEMENT
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'legend.title_fontsize': FONT_SIZE,
        'figure.titlesize': FONT_SIZE
    })

    models_in_data = results_df['Model'].unique()
    n_models = len(models_in_data)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]

    hue_order = ["CustomTfidf", "CustomBow", "TF-IDF", "BoW"]
    color_map = {
        "BoW": "#FF6666",         # Light red
        "CustomBow": "#CC0000",   # Dark red
        "TF-IDF": "#7CB5EC",
        "CustomTfidf": "#1F77B4"
    }

    for ax, model_label in zip(axes, models_in_data):
        df_model = results_df[results_df['Model'] == model_label].copy()

        sns.barplot(
            data=df_model,
            x='Macro F1-Score',
            y='Token Processing',
            hue='Representation',
            hue_order=hue_order,
            order=display_order,
            dodge=True,
            ax=ax,
            palette=color_map,
            alpha=0.95,
            edgecolor='white'
        )

        obecne_reprezentacje = [r for r in hue_order if r in df_model['Representation'].unique()]
        patch_idx = 0
        for rep in obecne_reprezentacje:
            for token_display in display_order:
                if patch_idx >= len(ax.patches):
                    break
                patch = ax.patches[patch_idx]
                patch_idx += 1

                row_matches = df_model[(df_model['Token Processing'] == token_display) &
                                       (df_model['Representation'] == rep)]
                if row_matches.empty:
                    continue
                row = row_matches.iloc[0]
                base_rep = 'BoW' if 'BoW' in rep or 'Bow' in rep else 'TF-IDF'
                baseline_filename = get_baseline_filename(model_label, base_rep).replace('.csv', '')
                comp_filename = row['Filename']
                p = p_dict.get((comp_filename, baseline_filename))
                if p is not None and p >= 0.05:
                    orig_color = patch.get_facecolor()
                    patch.set_facecolor('none')
                    patch.set_edgecolor(orig_color)
                    patch.set_hatch('////')

        for rep, score in baselines.get(model_label, {}).items():
            style = {'color': '#1F77B4' if rep == 'TF-IDF' else '#CC0000',
                     'ls': '--' if rep == 'TF-IDF' else ':'}
            ax.axvline(x=score, color=style['color'], linestyle=style['ls'],
                       linewidth=2.5, zorder=0)

        best_width = 0
        best_patch = None
        for patch in ax.patches:
            if patch.get_width() > best_width:
                best_width = patch.get_width()
                best_patch = patch

        if best_patch:
            ax.text(
                best_width + 0.001,
                best_patch.get_y() + best_patch.get_height() / 2,
                f"★ {best_width:.4f}",
                color='#333333',
                fontsize=FONT_SIZE,
                fontweight='bold',
                ha='left',
                va='center'
            )
            best_patch.set_edgecolor('#333333')
            best_patch.set_linewidth(2)

        x_min = max(0.0, global_min_f1 - 0.015)
        x_max = global_max_f1 + 0.015
        ax.set_xlim(x_min, x_max)
        ax.set_title(model_label, fontweight='bold', pad=15)
        ax.set_xlabel("Macro F1-Score", fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel("Preprocessing Step", fontweight='bold')
        else:
            ax.set_ylabel("")

        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.grid(axis='y', visible=False)

    for ax in axes:
        if ax.legend_:
            ax.legend_.remove()

    handles = [Patch(color=color_map[rep], label=rep) for rep in hue_order]
    handles.append(Patch(facecolor='none', edgecolor='#555555', hatch='////',
                         label='Insignificant (p >= 0.05)'))
    for rep in baseline_reps:
        style = {'color': '#1F77B4', 'ls': '--'} if rep == 'TF-IDF' else {'color': '#CC0000', 'ls': ':'}
        line = mlines.Line2D([], [], color=style['color'], linestyle=style['ls'], linewidth=2.5)
        handles.append(line)

    labels = [r for r in hue_order] + ['Insignificant (p >= 0.05)'] + \
             [f"{rep} Baseline (Expanded and Lower)" for rep in baseline_reps]

    fig.legend(handles=handles, labels=labels,
               loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0), frameon=False,
               fontsize=FONT_SIZE)

    sns.despine()
    plt.tight_layout()

    output_dir = Path("results/figures/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_study.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ablation study plot with optional McNemar significance shading.")
    parser.add_argument("--auto-run-mcnemar", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    parser.add_argument("--z-threshold", type=float, default=2.0, help="Z-score threshold used in file names (default: 2)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mcnemar-files", nargs="+", type=str)
    group.add_argument("--mcnemar-dir", type=str, default="results/analysis")

    args = parser.parse_args()

    mcnemar_files = None
    if args.auto_run_mcnemar:
        # Passed Z-threshold parameter to mcnemar func
        generated = run_mcnemar_tests(force=args.force, z_threshold=args.z_threshold)
        if generated:
            mcnemar_files = generated
    else:
        if args.mcnemar_files:
            mcnemar_files = args.mcnemar_files
        elif args.mcnemar_dir:
            dir_path = Path(args.mcnemar_dir)
            if dir_path.exists():
                mcnemar_files = list(dir_path.glob("*.csv"))

    # Passed Z-threshold parameter to plotting func
    plot_ablation_study(mcnemar_files=mcnemar_files, z_threshold=args.z_threshold)