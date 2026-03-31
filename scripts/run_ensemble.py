import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from src.utils.paths import PROJECT_ROOT, RESULTS_DIR

def save_cascade_results(y_true, hybrid_probs, name, split="test"):
    """Standardized saving logic matching BaseModel.evaluate()"""
    data_dir = RESULTS_DIR / split / "raw_predictions"
    report_dir = RESULTS_DIR / split / "classification_reports"

    for d in [data_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Save Raw Predictions (CSV)
    eval_df = pd.DataFrame({
        'true_label': y_true,
        'probability': hybrid_probs
    })
    eval_df.to_csv(data_dir / f"{name}.csv", index=False)

    # 2. Save Classification Report (CSV)
    preds = (hybrid_probs > 0.5).astype(int)
    report_dict = classification_report(y_true, preds, output_dict=True, digits=4)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_dir / f"{name}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.best_params) as f:
        cfg = yaml.safe_load(f)

    threshold = cfg.get('cascade', {}).get('delegation_threshold', 0.15)
    base_model_type = cfg.get('model', 'linear_svm')
    base_name = "Base_SVM" if base_model_type == 'linear_svm' else "Base_LogReg"
    
    pred_dir = RESULTS_DIR / "test" / "raw_predictions"
    
    # Load Base Data
    base_df = pd.read_csv(pred_dir / f"{base_name}.csv")
    y_true = base_df['true_label'].values
    p_base = base_df['probability'].values
    
    mask_uncertain = (p_base >= threshold) & (p_base <= (1.0 - threshold))
    
    specialists = {
        "SVM_Specialist": "SVM_Spec_Cascade",
        "BERT_Basic": "BERT_Basic_Cascade",
        "BERT_Specialist": "BERT_Spec_Cascade"
    }

    print(f"--- Processing Cascades (Threshold: {threshold}) ---")

    for spec_file, cascade_output_name in specialists.items():
        spec_path = pred_dir / f"{spec_file}.csv"
        
        if spec_path.exists():
            spec_df = pd.read_csv(spec_path)
            p_spec = spec_df['probability'].values

            # Create Hybrid Probability Array
            # We take base probs for certain samples, and specialist probs for uncertain ones
            hybrid_probs = p_base.copy()
            hybrid_probs[mask_uncertain] = p_spec[mask_uncertain]

            # Save in the same format as normal models
            save_cascade_results(y_true, hybrid_probs, cascade_output_name, split="test")
            print(f"Saved results for: {cascade_output_name}")
        else:
            print(f"Warning: {spec_file}.csv not found. Skipping.")

if __name__ == "__main__":
    main()