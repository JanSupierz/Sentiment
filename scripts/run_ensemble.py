import argparse
import yaml
import pandas as pd
from sklearn.metrics import f1_score

from src.utils.paths import PROJECT_ROOT, RESULTS_DIR


def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.best_params) as f:
        cfg = yaml.safe_load(f)

    threshold = cfg.get('cascade', {}).get('delegation_threshold', 0.15)
    base_model_type = cfg.get('model', 'linear_svm')

    base_name = "Base_SVM" if base_model_type == 'linear_svm' else "Base_LogReg"
    print(f"--- Simulating Cascade (Base Model: {base_name}) ---")
    print(f"Delegation Threshold: {threshold:.3f} (Lower) to {1.0 - threshold:.3f} (Upper)\n")
    pred_dir = RESULTS_DIR / "test" / "raw_predictions"

    paths = {
        "Base": pred_dir / f"{base_name}.csv",
        "SVM_Spec": pred_dir / "SVM_Specialist.csv",
        "BERT_Basic": pred_dir / "BERT_Basic.csv",
        "BERT_Spec": pred_dir / "BERT_Specialist.csv"
    }

    # Verify base predictions exist
    if not paths["Base"].exists():
        raise FileNotFoundError(f"Base predictions not found at {paths['Base']}. Run your train script first.")

    # Load Base Predictions & Define Delegation Mask
    base_df = pd.read_csv(paths["Base"])
    y_true = base_df['true_label'].values
    p_base = base_df['probability'].values
    pred_base = (p_base > 0.5).astype(int)

    # Find the data the base model is uncertain about
    mask_uncertain = (p_base >= threshold) & (p_base <= (1.0 - threshold))
    delegation_rate = mask_uncertain.mean()

    results = {
        "1. Just Base Model": calculate_f1(y_true, pred_base)
    }

    specialists = {
        "2. Base + Basic BERT": "BERT_Basic",
        "3. Base + Spec. BERT": "BERT_Spec",
        "4. Base + Spec. SVM": "SVM_Spec"
    }

    for label, spec_key in specialists.items():
        spec_path = paths[spec_key]
        if spec_path.exists():
            # Load specialist predictions
            spec_df = pd.read_csv(spec_path)
            p_spec = spec_df['probability'].values
            pred_spec = (p_spec > 0.5).astype(int)

            hybrid_preds = pred_base.copy()
            hybrid_preds[mask_uncertain] = pred_spec[mask_uncertain]

            results[label] = calculate_f1(y_true, hybrid_preds)
        else:
            print(f"Warning: {spec_path.name} not found. Skipping {label}.")

    # Save to CSV
    results_df = pd.DataFrame([results])
    results_df.insert(0, "Threshold", threshold)
    results_df.insert(1, "Delegation_Rate", delegation_rate)

    out_dir = RESULTS_DIR / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cascade_simulation_results.csv"
    results_df.to_csv(out_path, index=False)

    print("=== FINAL CASCADE COMPARISON (F1 SCORES) ===")
    print(f"Data Delegated to Specialist: {delegation_rate:.2%}\n")
    for name, f1 in results.items():
        print(f"{name:25s}: {f1:.2%}")

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()